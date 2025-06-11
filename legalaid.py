import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Load the PDF and extract text
loader = PyPDFLoader("Naija Constitutions.pdf")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128
)
split_docs = text_splitter.split_documents(docs)

# Extract content for embedding
doc_texts = [doc.page_content for doc in split_docs]

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(doc_texts, show_progress_bar=True)

# Streamlit UI
st.title("SUPERLEGAL - Ask About the Nigerian Constitution")
query = st.text_input("Ask a Question:")

if query:
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Get top 5 similar docs
    top_k = 5
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_docs = [doc_texts[i] for i in top_indices]

    def generate_response(query, relevant_texts):
        context = "\n".join(relevant_texts)
        prompt = f"""You are an experienced lawyer. Given the following context:\n\n{context}\n\nAnswer the question:\n{query}\n\nPlease respond conversationally."""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

    response = generate_response(query, relevant_docs)
    st.write(response)

