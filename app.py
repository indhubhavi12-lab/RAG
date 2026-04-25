import os
import streamlit as st
import google.generativeai as genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 🔐 Load API key
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# 🎯 Streamlit UI
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Chat with your PDF (Attention Paper)")

# 📂 Load PDF
pdf_path = "attention.pdf"

@st.cache_resource
def load_data():
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    db = FAISS.from_documents(documents, embeddings)

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=google_api_key
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa

qa_chain = load_data()

# 💬 Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question from the PDF:")

if user_input:
    response = qa_chain({"query": user_input})

    answer = response["result"]

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

# 🧠 Display chat
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")
