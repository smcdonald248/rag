"""
Author: Stephen McDonald
Description: A really simple RAG demo using Streamlit, OctoAI, and Pincone.io
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore

from helpers import Helpers

load_dotenv()


def return_llm(
    model: str,
    max_tokens: int,
    presence_penalty: float,
    temperature: float,
    top_p: float,
    streaming: bool
) -> ChatOpenAI:
    """ returns an OctoAI LLM Endpoint """
    return ChatOpenAI(
        base_url="https://text.octoai.run/v1/",
        api_key=os.getenv("OCTOAI_API_TOKEN"),
        model=model,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        temperature=temperature,
        top_p=top_p,
        streaming=streaming,
        stream_options={
            "include_usage":streaming
        }
    )


def return_template(rag: bool) -> str:
    """ returns the prompt template conditionally """
    if rag:
        return """
                    CONTEXT: {context}
                    QUESTION: {question}
                    SUPPORT FORM: https://creator.nightcafe.studio/feedback
                    
                    Instructions:
                    Answer the users QUESTION using the CONTEXT text above.
                    Keep your answer grounded in the facts of the CONTEXT.
                    Ignore the CONTEXT if it is not relevant and answer from
                    pre-trained knowledge.
                    Do not tell the user about the CONTEXT. If the context
                    is includes a link to the source of the retrieved context, then
                    include the link your response.
                    If the context includes instructions to contact support, provide
                    the link SUPPORT FORM that is listed above.
                """
    else:
        return """
                    QUESTION: {question}            
        
                    Instructions:
                    Below is QUESTION that describes a task or a general ask. 
                    Write a response that appropriately completes the request.\n 
                """


def get_chain(
    llm: ChatOpenAI,
    rag: bool,
    prompt: PromptTemplate,
    retriever: PineconeVectorStore = None,
) -> LLMChain:
    """Init"""

    if rag:
        chain: Runnable = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        chain: Runnable = prompt | llm #| StrOutputParser()
    return chain


def exec_llm(form_question: str, llm_chain: LLMChain, streaming: bool) -> str:
    """execute the llm chain"""
    if streaming:
        for event in llm_chain.stream(form_question):
            os.write(1, f"{event}".encode())
        return llm_chain.stream(form_question)
    else:
        return llm_chain.invoke(form_question)


def load_vector_store(data: str, embedding: OctoAIEmbeddings, index: str) -> None:
    """load vector database with data from documentation url"""
    PineconeVectorStore.from_documents(data, embedding, index_name=index)

@st.experimental_fragment
def add_data_to_index(url: str, embedding: OctoAIEmbeddings, index: str):
    """ add data to a pinecone index """
    with st.status("Scanning URL for links...") as status:
        helpers = Helpers()
        helpers.get_links(url, url, lvl=4)
        st.write("Extracting and Partitioning Data...")
        helpers.get_data()
        st.write(f"Loading VectorDb Index: {index}")
        load_vector_store(helpers.agg_chunks, embedding, index)
        status.update(label=f"Data load into {index} complete!", state="complete")

def main() -> None:
    """app entrypoint"""
    embed: OctoAIEmbeddings = OctoAIEmbeddings(
        endpoint_url="https://text.octoai.run/v1/embeddings",
        octoai_api_token=os.getenv("OCTOAI_API_TOKEN"),
    )
    models: list = []
    # Adding new models that aren't yet listed as available
    models.append("mixtral-8x7b-instruct")
    models.append("meta-llama-3.1-8b-instruct")
    models.append("meta-llama-3.1-70b-instruct")
    models.append("meta-llama-3.1-405b-instruct")

    st.title("RAG Demo")
    with st.sidebar:
        st.header("Options")

        model: str = st.selectbox("Model", options=models, index=models.index("meta-llama-3.1-8b-instruct"))
        max_tokens: int = st.slider("max_tokens", 128, 8192, 4096)
        presence_penalty: float = st.slider("presence_penalty", 0.0, 1.0, 0.0)
        temperature: float = st.slider("temperature", 0.0, 2.0, 0.0)
        top_p: float = st.slider("top_p", 0.0, 1.0, 1.0)
        streaming: bool = st.toggle("Streaming")
        st.divider()

        enable_rag: bool = st.toggle("Enable RAG")
        if enable_rag:
            st.header("RAG Options")
            index_name = st.text_input("index_name", "octo")
            score = st.slider("similarity_score_threshold", 0.0, 1.0, 0.9)
            st.subheader("Add More Context")
            st.text_input("url", key="ingest")

            if "ingest" in st.session_state:
                if st.button("Scan"):
                    add_data_to_index(
                        url=st.session_state.ingest,
                        embedding=embed,
                        index=index_name
                    )

    llm: ChatOpenAI = return_llm(
        model=model,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        temperature=temperature,
        top_p=top_p,
        streaming=streaming
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
            response = exec_llm(prompt, chain, streaming)
            if streaming:
                response = st.write_stream(response)
                os.write(1 , f"{response}".encode())
                st.markdown(response.response_metadata["token_usage"])
            else:
                st.markdown(response.content)
                st.markdown(response.response_metadata["token_usage"])
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
