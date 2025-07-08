import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
openai.openai_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(
                page_title = "Chat with you Documents, powered by LlamaIndex",
                layout = "centered",
                initial_sidebar_state = "auto",
                menu_items = None
                )
st.title("Chat with your Documents")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Pakistan Studies!"
        }
    ]
@st.cache_resource(show_spinner=False)


def load_data():
    with st.spinner(text = "Loading and Indexing the Documents...."):
        reader = SimpleDirectoryReader(
            input_dir = "data",
            recursive = True
        )
        docs = reader.load_data()
        embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        llm = OpenAI(
            model = "gpt-3.5-turbo",
            temperature = "0.5",
            systemprompt = """You are expert in Pakistan Studies.
                              Your job is provide the valid and relevant answers.
                              Assume that all the queries are related to Pakistan Studies.
                              Keep your answers based on facts and do not hallucinate."""
        )
        Settings.llm = llm
        Settings.embed_model = embedding_model
        index = VectorStoreIndex.from_documents(docs)
        return index


index = load_data()
chat_engine = index.as_chat_engine(chat_mode = "condense_question", verbose = True)
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {
                "role": "assistant",
                "content": response.response
            }
            st.session_state.messages.append(message)
