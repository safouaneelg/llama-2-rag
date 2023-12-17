from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_transformers import Html2TextTransformer
import torch
import nest_asyncio

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# DATA_PATH = 'data/'
# DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def vectordb():
    
    with open("./data/links_2_treeblogs.txt", 'r') as file:
        links = file.read().splitlines()
    
    nest_asyncio.apply()
    
    loader = AsyncChromiumLoader(links) 

    documents = loader.load()

    html2txt = Html2TextTransformer()
    documents = html2txt.transform_documents(documents)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceInstructEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': DEVICE})
    
    db = Chroma.from_documents(texts, embeddings, persist_directory="db")

    """ db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH) """

if __name__ == "__main__":
    vectordb()

