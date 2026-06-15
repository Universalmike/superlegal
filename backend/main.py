import os
import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# Setup environment
load_dotenv()

app = FastAPI(title="Super Legal Backend API", version="1.0.0")

# CORS middleware to allow the React frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEXES_DIR = os.path.join(BASE_DIR, "backend", "indexes")
CSV_LOG_FILE = os.path.join(BASE_DIR, "backend", "chat_log.csv")

DOCUMENT_OPTIONS = {
    "Nigerian Constitution": os.path.join(BASE_DIR, "documents", "Naija Constitutions.pdf"),
    "Labour Law Act": os.path.join(BASE_DIR, "documents", "LABOUR_ACT.pdf"),
    "Criminal Code": os.path.join(BASE_DIR, "documents", "C38.pdf")
}

# In-memory storage for sessions: { session_id: [ {"sender": "user"|"ai", "text": str}, ... ] }
chat_sessions: Dict[str, List[Dict[str, str]]] = {
    "Chat 1": []
}

# Cached components in-memory
embedding_model = None
vector_stores = {}

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        # Load HuggingFace embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def load_vector_store(doc_key: str):
    if doc_key not in DOCUMENT_OPTIONS:
        raise ValueError(f"Document key '{doc_key}' not found.")
        
    if doc_key in vector_stores:
        return vector_stores[doc_key]
        
    file_path = DOCUMENT_OPTIONS[doc_key]
    store_name = doc_key.replace(" ", "_").lower()
    faiss_dir = os.path.join(INDEXES_DIR, store_name)
    faiss_index = os.path.join(faiss_dir, "index.faiss")
    faiss_pkl = os.path.join(faiss_dir, "index.pkl")
    
    embedder = get_embedding_model()
    
    if os.path.exists(faiss_index) and os.path.exists(faiss_pkl):
        vector_db = FAISS.load_local(faiss_dir, embedder, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source PDF not found at {file_path}")
        os.makedirs(faiss_dir, exist_ok=True)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        vector_db = FAISS.from_documents(split_docs, embedder)
        vector_db.save_local(faiss_dir)
        
    vector_stores[doc_key] = vector_db
    return vector_db

def log_to_csv(session_name, document, language, question, answer):
    file_exists = os.path.exists(CSV_LOG_FILE)
    os.makedirs(os.path.dirname(CSV_LOG_FILE), exist_ok=True)
    with open(CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "session", "document", "language", "question", "answer"])
        writer.writerow([datetime.now().isoformat(), session_name, document, language, question, answer])

class ChatRequest(BaseModel):
    session_id: str
    document_key: str
    language: str
    query: str
    api_key: Optional[str] = None

class CreateSessionRequest(BaseModel):
    session_id: str

@app.get("/api/documents")
def get_documents():
    return list(DOCUMENT_OPTIONS.keys())

@app.get("/api/sessions")
def get_sessions():
    return list(chat_sessions.keys())

@app.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    session_id = req.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")
    if session_id in chat_sessions:
        raise HTTPException(status_code=400, detail="Session already exists.")
    chat_sessions[session_id] = []
    return {"status": "success", "session_id": session_id}

@app.get("/api/sessions/{session_id}")
def get_session_history(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return chat_sessions[session_id]

@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        
    # Configure API Key (prefer request parameter, fallback to env)
    api_key = req.api_key or os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400, 
            detail="Gemini API Key is missing. Please set it in your environment or provide it in the chat interface settings."
        )
    
    try:
        genai.configure(api_key=api_key)
        vector_db = load_vector_store(req.document_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")
        
    try:
        # Perform retrieval
        retriever = vector_db.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(req.query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Build conversational history for context (last 3 turns / 6 messages)
        history_str = ""
        recent_messages = chat_sessions[session_id][-6:]
        for msg in recent_messages:
            role = "Client" if msg["sender"] == "user" else "Assistant"
            history_str += f"{role}: {msg['text']}\n"
            
        prompt = f"""You are an experienced Nigerian legal assistant.
Conversation History:
{history_str}

The client has asked: {req.query}

Relevant legal context from {req.document_key}:
{context}

Explain the key legal points clearly, as you would in a client consultation. Refer to specific sections or articles from the text if available.
Respond in {req.language} only."""

        # Generate response using gemini-2.5-flash
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        result = model.generate_content(prompt)
        answer = result.text
        
        # Save to chat history
        chat_sessions[session_id].append({"sender": "user", "text": req.query})
        chat_sessions[session_id].append({"sender": "ai", "text": answer})
        
        # Log to CSV
        log_to_csv(session_id, req.document_key, req.language, req.query, answer)
        
        return {
            "answer": answer,
            "session_id": session_id,
            "references": [
                {"page_content": doc.page_content, "metadata": getattr(doc, "metadata", {})} 
                for doc in docs[:3]  # Return top 3 references for display
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
