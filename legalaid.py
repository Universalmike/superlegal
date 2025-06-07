import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# Load the PDF and extract text
loader = PyPDFLoader("Naija Constitutions.pdf")
docs = loader.load()

# Increase chunk size and add overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,  # Increase to 1000-1500 for more context
    chunk_overlap=128  # Overlap to maintain context across chunks
)

# Split documents into chunks
split_docs = text_splitter.split_documents(docs)

# Use an efficient embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in ChromaDB
vector_db = Chroma.from_documents(split_docs, embedding_model, persist_directory="db")
vector_db.persist()

retriever = vector_db.as_retriever(search_kwargs={"k": 15})  # Retrieve top 3 most relevant chunk

load_dotenv()

#Retrieve api key
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

st.write("SUPERLEGAL")
#query = "What are the powers of the state?"
query = st.text_input("Ask a Question: ")
retrieved_docs = retriever.get_relevant_documents(query)

def generate_response(query, relevant_documents):
    context = "\n".join([doc.page_content for doc in relevant_documents])  # Combine the relevant documents into context
    prompt = f"""You are a experienced lawyer, Given the following information \n{context}\nAnswer the following question as an experienced lawyer:\n{query},
    ,answer me like we are in a conversation"""

    # Construct prompt based on the row data
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

response = generate_response(query, retrieved_docs)
st.write(response)
