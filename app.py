"""
Author: Stephen McDonald
Description: A really simple RAG demo using OctoAI and Milvus vector database
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
import octoai

from helpers import Helpers

load_dotenv()


def return_llm(
    model: str,
    max_tokens: int,
    presence_penalty: float,
    temperature: float,
    top_p: float,
) -> OctoAIEndpoint:
    return OctoAIEndpoint(
        endpoint_url="https://text.octoai.run/v1/chat/completions",
        model_kwargs={
            "model": model,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep your responses limited to one short paragraph if possible.",
                },
            ],
        },
    )


def return_template(rag: bool) -> str:
    if rag:
        return """
                    CONTEXT: {context}
                    QUESTION: {question}

                    Instructions:
                    Answer the users QUESTION using the CONTEXT text above.
                    Keep your answer grounded in the facts of the CONTEXT.
                    Ignore the CONTEXT if it is not relevant and answer from
                    pre-trained knowledge.
                    Do not tell the user that about the CONTEXT.
                """
    else:
        return """
                    QUESTION: {question}            
        
                    Instructions:
                    Below is QUESTION that describes a task or a general ask. 
                    Write a response that appropriately completes the request.\n 
                """


def get_chain(
    llm: OctoAIEndpoint,
    rag: bool,
    prompt: PromptTemplate,
    retriever: PineconeVectorStore = None,
) -> LLMChain:
    """Init"""

    if rag:
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        chain = prompt | llm | StrOutputParser()
    return chain


def exec_llm(form_question: str, llm_chain: LLMChain) -> str:
    """execute the llm chain"""
    for chunk in llm_chain.stream(form_question):
        yield chunk
    # return llm_chain.invoke(form_question) #.get("text")


def load_vector_store(data: str, embedding: OctoAIEmbeddings, index: str) -> None:
    """load vector database with data from documentation url"""
    PineconeVectorStore.from_documents(data, embedding, index_name=index)

    return

@st.experimental_fragment
def add_data_to_index(url: str, embedding: OctoAIEmbeddings, index: str):
    with st.status("Scanning URL for links...") as status:
        helpers = Helpers()
        helpers.get_links(url, url)
        st.write("Extracting and Partitioning Data...")
        helpers.get_data()
        st.write(f"Loading VectorDb Index: {index}")
        load_vector_store(helpers.agg_chunks, embedding, index)
        status.update(f"Data load into {index} complete!")

def main() -> None:
    """app entrypoint"""
    embed: OctoAIEmbeddings = OctoAIEmbeddings(
        endpoint_url="https://text.octoai.run/v1/embeddings",
        octoai_api_token=os.getenv("OCTOAI_API_TOKEN"),
    )

    st.title("RAG Demo")
    with st.sidebar:
        st.header("Menu")
        enable_rag: bool = st.toggle("Enable RAG")

        if enable_rag:
            index_name = st.text_input("index_name", "rag-demo")
            score = st.slider("similarity_score_threshold", 0.0, 1.0, 0.9)
        model: str = st.selectbox("Model", octoai.chat.get_model_list())
        max_tokens: int = st.slider("max_tokens", 128, 1024, 512)
        presence_penalty: float = st.slider("presence_penalty", 0.0, 0.0, 1.0)
        temperature: float = st.slider("temperature", 0.0, 2.0, 0.75)
        top_p: float = st.slider("top_p", 0.0, 1.0, 0.95)
        st.divider()
        st.subheader("Add More Context")
        st.text_input("url", key="ingest")
        if "ingest" in st.session_state:
            os.write(0, "url in session state".encode())
            if st.button("Scan"):
                os.write(0, "downloading context".encode())
                add_data_to_index(
                    url=st.session_state.ingest,
                    embedding=embed,
                    index=index_name
                )

    llm = return_llm(
        model=model,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        temperature=temperature,
        top_p=top_p,
    )
    prompt: PromptTemplate = PromptTemplate.from_template(
        return_template(rag=enable_rag)
    )
    if enable_rag:
        pcvs = PineconeVectorStore(index_name=index_name, embedding=embed)
        chain = get_chain(
            rag=enable_rag,
            llm=llm,
            prompt=prompt,
            retriever=pcvs.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score},
            ),
        )
    else:
        chain = get_chain(rag=enable_rag, llm=llm, prompt=prompt)

    try:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(""):
            st.session_state.messages.append({"role": "human", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = st.write_stream(exec_llm(prompt, chain))
            st.session_state.messages.append({"role": "ai", "content": response})

    except Exception as e:
        os.write(1, str(e).encode())


if __name__ == "__main__":
    main()
