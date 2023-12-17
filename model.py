from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from auto_gptq import AutoGPTQForCausalLM
import chainlit as cl
import torch
from transformers import AutoTokenizer, TextStreamer, pipeline

# # # # First check if cuda is available otherwise just use cpu
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# # # # This is a list of pre-prompts
SYSTEM_PROMPT = """
- Your name is Saf-Agent.
- Use the following pieces of information to answer the user's question.
- If you don't know the answer, just say that you don't know and don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# # # # Loading the model
def load_llm():
    # I've selected quantized Llama2 7B finetuned on instruction. Other models could be used !
    model_name_or_path = "TheBloke/Llama-2-7B-Chat-GPTQ"
    mode_basename = "model"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        revision="gptq-4bit-128g-actorder_True",
        model_basename=mode_basename,
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device=DEVICE,
        quantize_config=None,
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.15,
        streamer=streamer,
    )
    # Load the LLM model here
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
    return llm

# # # # QA Model Function
def qa_ai_chat():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': DEVICE})
    db = Chroma(persist_directory='./db', embedding_function=embeddings)
    llm = load_llm()

    # Prompt template for QA retrieval for each vectorstore
    qa_prompt = PromptTemplate(template=SYSTEM_PROMPT,
                            input_variables=['context', 'question'])

    #Retrieval QA Chain
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=db.as_retriever(search_kwargs={'k': 2, 'threshold':0.5}),
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': qa_prompt}
                                    )

    return qa

#output function
def final_result(query):
    qa_result = qa_ai_chat()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_ai_chat()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, I'm Saf-Agent. How can I help?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"][0].metadata 

    if sources:
        answer += f"\n" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

