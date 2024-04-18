"""
Author: Stephen McDonald
Description: A really simple RAG demo using OctoAI and Milvus vector database
"""

import os
import subprocess
import streamlit as st
from typing import Any
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from milvus import default_server
from helpers import Helpers


def initiate() -> LLMChain:
    """Init"""
    load_dotenv()
    subprocess.run(["playwright", "install"])
    subprocess.run(["sudo", "apt-get", "install", "libnss3", "libnspr4", "libatk1.0-0", "libatk-bridge2.0-0", "libcups2", "libdrm2", "libatspi2.0-0", "libxcomposite1", "libxdamage1", "libxfixes3", "libxrandr2", "libgbm1", "libxkbcommon0", "libpango-1.0-0", "libcairo2", "libasound2"])

    template: str = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n Instruction:\n{form_question}\n Response: """
    prompt: PromptTemplate = PromptTemplate.from_template(template)

    llm: OctoAIEndpoint = OctoAIEndpoint(
        endpoint_url="https://text.octoai.run/v1/chat/completions",
        model_kwargs={
            "model": "mixtral-8x7b-instruct-fp16",
            "max_tokens": 256,
            "presence_penalty": 0,
            "temperature": 0.75,
            "top_p": 0.95,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep your responses limited to one short paragraph if possible.",
                },
            ],
        },
    )

    llm_chain: LLMChain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

def rag_initiate(retriever) -> LLMChain:
    """Init"""
    load_dotenv()

    template: str = """Answer the question based on the following context. 
                        Context: {context}
                        Question: {question}
                    """
    prompt: PromptTemplate = PromptTemplate.from_template(template)

    llm: OctoAIEndpoint = OctoAIEndpoint(
        endpoint_url="https://text.octoai.run/v1/chat/completions",
        model_kwargs={
            "model": "mixtral-8x7b-instruct-fp16",
            "max_tokens": 256,
            "presence_penalty": 0,
            "temperature": 0.75,
            "top_p": 0.95,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep your responses limited to one short paragraph if possible.",
                },
            ],
        },
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def exec_llm(form_question: str, llm_chain: LLMChain) -> str:
    """execute the llm chain"""
    return llm_chain.invoke(form_question) #.get("text")

def load_vectors(data: str) -> Any:
    """load vector database with data from documentation url"""
    default_server.start()
    
    # Use an OctoAI Embedding
    embed = OctoAIEmbeddings(
        endpoint_url="https://text.octoai.run/v1/embeddings",
        octoai_api_token=os.getenv('OCTOAI_API_TOKEN')
    )

    # Store the vectors in Milvus vector DB - takes a couple of minutes
    vector_store = Milvus.from_documents(
        data,
        embedding=embed,
        connection_args={"host": '127.0.0.1', "port": default_server.listen_port},
        collection_name="Documentation"
    )

    return vector_store.as_retriever()

def main(llm_chain: LLMChain) -> None:
    """app entrypoint"""
    try:
        st.header("RAG Demo")

        st.text_input("URL to documentation", key="url")
        st.text_input("Question", key="form_question")
        if st.button("Submit", key="question_button"):
            if st.session_state.url:
                helpers = Helpers()
                helpers.get_links(st.session_state.url, st.session_state.url, 1)
                helpers.get_data()
                chain = rag_initiate(load_vectors(helpers.agg_chunks))
                st.write("Context Loading Completed.")
                os.write(1, b"In the RAG Chain")
                st.write(exec_llm(st.session_state.form_question, chain))
            else:
                os.write(1, b"Not in the RAG Chain")
                st.write(exec_llm(st.session_state.form_question, llm_chain))
    except Exception as e:
        os.write(1, str(e).encode())
    finally:
        default_server.stop()

if __name__ == "__main__":
    main(initiate())