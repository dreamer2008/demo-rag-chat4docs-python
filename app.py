import os
from enum import Enum

import docx
import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from htmlTemplates import css, bot_template, user_template

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# define an enumeration for LLM type
class ModelType(Enum):
    GOOGLE_GENAI = 1,
    HUGGINGFACE = 2,
    OPEN_AI = 3,
    OLLAMA = 4


def main():
    load_dotenv()
    model_type = ModelType.HUGGINGFACE
    st.set_page_config(page_title="Chat with documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Welcome to the ChatBot for documents")
    user_input = st.text_input("Ask me anything about your documents")
    if user_input:
        handle_user_input(user_input)

    with st.sidebar:
        st.subheader("Your Documents")
        docs = st.file_uploader("Upload a file and click on 'Process'", type=["pdf", "txt", "docx"], key="file_upload",
                                accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # read the file and process it
                raw_text = get_doc_text(docs)
                print(len(raw_text))

                # get the chunks
                chunks = get_text_chunks(raw_text)
                # st.write(f"{len(chunks)} chunks created")
                st.write(chunks)

                # get the embeddings
                vector_store = get_vector_store(chunks, model_type)
                print("vs *******")
                print(vector_store)
                st.session_state.conversation = get_chat_chain(vector_store, model_type)
                st.success("Done")


def get_doc_text(docs):
    text = ""
    for doc in docs:
        filename, extension = doc.name.split('.')
        # print(f"{extension}")
        if "pdf" == extension:
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif "txt" == extension:
            text += doc.read().decode("utf-8")
        elif "docx" == extension:
            # text += doc.read().decode("utf-8")
            doc = docx.Document(doc)
            for paragraph in doc.paragraphs:
                text += paragraph.text
    return text


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(chunks, model_type):
    if model_type == ModelType.OPEN_AI:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            deployment="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL")
        )
    elif model_type == ModelType.HUGGINGFACE:
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def get_llm_model(model_type):
    if model_type == ModelType.GOOGLE_GENAI:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    elif model_type == ModelType.HUGGINGFACE:
        llm = HuggingFaceHub(repo_id="google/flan-t5-base")
    elif model_type == ModelType.OPEN_AI:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL")
        )
    else:
        # llm = OllamaLLM(model="llama3.2:latest")
        llm = OllamaLLM(model="qwen2.5:7b")
    return llm


def get_chat_chain(vector_store, model_type):
    llm = get_llm_model(model_type)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=True
    )
    return conversation_chain


def handle_user_input(user_input):
    # response = st.session_state.conversation({"question": user_input})
    response = st.session_state.conversation.invoke({"question": user_input})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
