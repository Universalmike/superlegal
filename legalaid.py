# Multi-Document Legal Chat App with Session-based Chat History (WhatsApp Style)

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
    "Nigerian Constitution": "Naija Constitutions.pdf",
    "Labour Law Act": "LABOUR_ACT.pdf",
    "Criminal Code": "C38.pdf"
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
        return FAISS.load_local(faiss_dir, embedding_model, allow_dangerous_deserialization=True)
    else:
        os.makedirs(faiss_dir, exist_ok=True)
        vector_db = FAISS.from_documents(split_docs, embedding_model)
        vector_db.save_local(faiss_dir)
        return vector_db

# Sidebar: document and chat management
st.sidebar.title("📚 Legal Assistant Settings")
selected_doc = st.sidebar.selectbox("Choose a Legal Aspect", list(DOCUMENT_OPTIONS.keys()))

if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'active_chat' not in st.session_state:
    st.session_state.active_chat = "Chat 1"

# Create new chat session
new_chat = st.sidebar.text_input("Start New Chat", key="new_chat")
if st.sidebar.button("Create Chat") and new_chat.strip():
    st.session_state.chats[new_chat] = []
    st.session_state.active_chat = new_chat

# Select existing chat
chat_names = list(st.session_state.chats.keys()) or ["Chat 1"]
selected_chat = st.sidebar.selectbox("Active Chat", chat_names, index=chat_names.index(st.session_state.active_chat))
st.session_state.active_chat = selected_chat

# Initialize chat history if not already
if selected_chat not in st.session_state.chats:
    st.session_state.chats[selected_chat] = []

# Load vector DB for selected document
doc_path = DOCUMENT_OPTIONS[selected_doc]
doc_name = selected_doc.replace(" ", "_").lower()
vector_db = load_vector_store(doc_path, doc_name)

# UI Layout for WhatsApp-style chat
st.title(f"📖 Ask D Law – {selected_doc}")
language = st.selectbox("Choose response language:", ["English", "French", "Yoruba", "Hausa", "Igbo"])

st.markdown("""
    <style>
        .chat-bubble-user {
            background-color: #dcf8c6;
            padding: 0.5em;
            margin: 0.3em;
            border-radius: 0.5em;
            text-align: right;
        }
        .chat-bubble-ai {
            background-color: #ffffff;
            padding: 0.5em;
            margin: 0.3em;
            border-radius: 0.5em;
            border: 1px solid #ccc;
            text-align: left;
        }
        .chat-box {
            max-height: 500px;
            overflow-y: auto;
            margin-bottom: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Display chat history (bottom aligned)
st.markdown("### 💬 Chat")
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
for user_q, ai_a in st.session_state.chats[selected_chat]:
    st.markdown(f"<div class='chat-bubble-user'>🧑‍⚖️ You: {user_q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-bubble-ai'>🤖 D Law: {ai_a}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Question input below chat history
query = st.text_area("Type your question...", height=100)

# Generate response
if st.button("Send") and query:
    retriever = vector_db.as_retriever(search_kwargs={"k": 20})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an experienced Nigerian legal assistant.
    I have a legal question regarding {query}  Use the context {context} to answer the question:
    Imagine you're explaining this to a client in a consultation. What are the key points I should be aware of?
    Respond clearly and accurately in {language} only.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content(prompt)
    answer = result.text

    # Save to session
    st.session_state.chats[selected_chat].append((query, answer))

    # Refresh UI (optional to simulate bottom-scroll)
    st.experimental_rerun()

for user_q, ai_a in st.session_state.sessions[selected_session]:
    st.markdown(f"**You:** {user_q}")
    st.markdown(f"**D Law:** {ai_a}")
    st.markdown("---")

