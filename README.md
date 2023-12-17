

# Saf-AI: RETRIEVAL AUGMENTED GENERATION FROM LINKS USING LLAMA2 - 

This Llama2 AI is able to answer question based on a list of webpage links. It is a chat tool able to provide information about Canada Plum while giving also the sources from where the info has been retrieved.

Thanks to the use of quantized Llama2-7b, `"TheBloke/Llama-2-7B-Chat-GPTQ"` It is able to run on very low GPU devices and/or CPUs. If you hardware can support better models, you can try `"TheBloke/llama-2-13b-chat-gptq"` or even `"TheBloke/llama-2-70b-chat-gptq"`. Just change the variable ```model_name_or_path``` in model.py. 

This README will guide you through the setup and usage of the Llama2 chatbot.

Demo:
![demo](https://github.com/safouane95/llama-2-rag/assets/54261127/c5bdaeb8-6732-47be-8699-b467e3c2cffd)


## Table of Contents

- [Introduction](#LLAMA2-RAG)
- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [License](#license)

## Prerequisites

This LLAMA2 bot has been built using the following configuration

- Python 3.10
- Required Python packages (you can install them using pip):
    - langchain
    - chainlit
    - sentence-transformers
    - chromedb
    - html2text

## Installation

1. Clone this repository to your local machine.

    ```bash
    git clone https://github.com/safouane95/llama-2-rag.git
    cd llama-2-rag
    ```

2. Create a Python virtual environment:

    ```bash
    conda create --name SafAIlab python=3.10 -y
    conda activate SafAIlab
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. change the links in the links_2_treeblogs.txt file .

## Getting Started

To get started with the Llama2 chatbot, follow those steps:

1. Once the packages installed, run the following command to create vector db path `./db`:

    ```bash
    python create_vectordb.py
    ```
Depending on your hardware setup, the given links and the number of the pages, this might takes few seconds/minutes.

2. To run the chainlit demo app run this code:

    ```bash
    chainlit run model.py -w
    ```
This command will make your app available at http://localhost:8000
(ensure this port isn't taken by another process or change the port)

3. Enjoy your chitchat with the arborist AI


## License

This project is licensed under the MIT License.

---

For more information on how to use, configure, and extend the Llama2 chatbot, please refer to the Langchain documentation or contact the project maintainers.
