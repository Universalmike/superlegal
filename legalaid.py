import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Path to FAISS index
faiss_path = "faiss_index"

# Load or build FAISS
if os.path.exists(faiss_path):
    vector_db = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
else:
    loader = PyPDFLoader("Naija Constitutions.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    split_docs = text_splitter.split_documents(docs)
    vector_db = FAISS.from_documents(split_docs, embedding_model)
    vector_db.save_local(faiss_path)

# Streamlit UI
st.title("SUPERLEGAL - Ask About the Nigerian Constitution")
query = st.text_input("Ask a Question:")

if query:
    retriever = vector_db.as_retriever(search_kwargs={"k": 15})  # Slightly more for expansion
    retrieved_docs = retriever.get_relevant_documents(query)

    # Query Expansion: add semantic context from top retrieved chunks
    expansion_text = "\n".join([doc.page_content[:300] for doc in retrieved_docs[:9]])  # Use top 3 for expansion

    def generate_response(original_query, expansion):
        expanded_query = f"{original_query}\n\nAdditional related context:\n{expansion}"
        prompt = f"""
        You are an experienced constitutional lawyer. Given the following expanded information and question:\n\n{expanded_query}\n
        Answer clearly like we’re having a conversation.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    response = generate_response(query, expansion_text)
    st.write(response)

