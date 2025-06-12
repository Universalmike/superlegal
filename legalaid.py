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

# Embedding model (reused whether loading or creating FAISS)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Path to FAISS index directory
faiss_path = "faiss_index"

# Load or build FAISS index
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
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    retrieved_docs = retriever.get_relevant_documents(query)

    def generate_response(query, relevant_docs):
        context = "\n".join([doc.page_content[:500] for doc in relevant_docs])
        prompt = f"""You are an experienced lawyer. Given the following context:\n\n{context}\n\nAnswer this question:\n{query}
                    \n\nRespond like we’re in a conversation."""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    response = generate_response(query, retrieved_docs)
    st.write(response)

