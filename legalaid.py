# Multi-Document Legal Chat App with Session-based Chat History

import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai

# Initialize environment
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

# Embedding model (cached)
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedder()

# Available legal documents
DOCUMENT_OPTIONS = {
    "Nigerian Constitution": "documents/Naija Constitutions.pdf",
    "Labour Law Act": "documents/LABOUR_ACT.pdf"
    #"Evidence Act": "Evidence Act.pdf"
}

# Load and embed selected document (with caching)
@st.cache_resource
def load_vector_store(file_path, store_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    faiss_dir = os.path.join("indexes", store_name)
    faiss_index = os.path.join(faiss_dir, "index.faiss")
    faiss_pkl = os.path.join(faiss_dir, "index.pkl")

    if os.path.exists(faiss_index) and os.path.exists(faiss_pkl):
        return FAISS.load_local(faiss_dir, embedding_model)
    else:
        os.makedirs(faiss_dir, exist_ok=True)
        vector_db = FAISS.from_documents(split_docs, embedding_model)
        vector_db.save_local(faiss_dir)
        return vector_db

# Sidebar: document and session selection
st.sidebar.title("📚 Legal Assistant Settings")
selected_doc = st.sidebar.selectbox("Choose a Legal Aspect", list(DOCUMENT_OPTIONS.keys()))

if 'sessions' not in st.session_state:
    st.session_state.sessions = {}
if 'current_session' not in st.session_state:
    st.session_state.current_session = "Session 1"

# New session input
new_session_name = st.sidebar.text_input("Start New Chat Session", key="new_session")
if st.sidebar.button("Create Session") and new_session_name.strip():
    st.session_state.sessions[new_session_name] = []
    st.session_state.current_session = new_session_name

# Switch between sessions
session_names = list(st.session_state.sessions.keys()) or ["Session 1"]
selected_session = st.sidebar.selectbox("Active Session", session_names, index=session_names.index(st.session_state.current_session))
st.session_state.current_session = selected_session

# Initialize chat history for session
if selected_session not in st.session_state.sessions:
    st.session_state.sessions[selected_session] = []

# Load selected document index
doc_path = DOCUMENT_OPTIONS[selected_doc]
doc_name = selected_doc.replace(" ", "_").lower()
vector_db = load_vector_store(doc_path, doc_name)

# UI title and language choice
st.title(f"📖 Ask D Law – {selected_doc}")
language = st.selectbox("Choose response language:", ["English", "French", "Yoruba", "Hausa", "Igbo"])
query = st.text_input("Ask a question:")

# Generate response
if query:
    retriever = vector_db.as_retriever(search_kwargs={"k": 15})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an experienced Nigerian legal assistant. Use the context below to answer the question:

    Context:
    {context}

    Question:
    {query}

    Translate the answer into {language} at the end.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content(prompt)
    answer = result.text

    # Save to chat session
    st.session_state.sessions[selected_session].append((query, answer))

# Show chat history like WhatsApp style
st.markdown("---")
st.subheader(f"💬 Chat – {selected_session}")
for user_q, ai_a in st.session_state.sessions[selected_session]:
    st.markdown(f"**You:** {user_q}")
    st.markdown(f"**D Law:** {ai_a}")
    st.markdown("---")

