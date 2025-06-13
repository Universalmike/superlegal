import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Load and split documents
loader = PyPDFLoader("Naija Constitutions.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128
)
split_docs = text_splitter.split_documents(docs)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create or load FAISS vector store
if not os.path.exists("faiss_index"):
    vector_db = FAISS.from_documents(split_docs, embedding_model)
    vector_db.save_local("faiss_index")
else:
    vector_db = FAISS.load_local("faiss_index", embedding_model)

# Streamlit UI
st.title("🧠 SUPERLEGAL RAG System")
query = st.text_input("Ask a legal question:")

# Language selector
language = st.selectbox("Choose response language:", ["English", "French", "Yoruba", "Hausa", "Igbo"])

# Function to generate response
def generate_response(query, relevant_documents, language):
    context = "\n".join([doc.page_content for doc in relevant_documents])
    prompt = f"""You are an experienced Nigerian legal assistant. Use the context below to answer the question as if in a conversation:
    
Context:
{context}

Question:
{query}

Respond clearly and accurately. Translate the answer into {language} at the end.
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Handle query
if query:
    retriever = vector_db.as_retriever(search_kwargs={"k": 15})
    retrieved_docs = retriever.get_relevant_documents(query)

    response = generate_response(query, retrieved_docs, language)

    # Show response
    st.markdown("### 💬 Response")
    st.write(response)

    # Save to chat history
    st.session_state.chat_history.append((query, response))

# Show chat history
if st.session_state.chat_history:
    st.markdown("### 🕓 Chat History")
    for i, (q, r) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {r}")
        st.markdown("---")
