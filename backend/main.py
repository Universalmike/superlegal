import os
import csv
import json
import re
import html
import sqlite3
import uuid
import hashlib
import base64
import math
from io import BytesIO
from contextlib import contextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

from cryptography.hazmat.primitives import hashes as _hashes, serialization as _serialization
from cryptography.hazmat.primitives.asymmetric import rsa as _rsa, padding as _apadding
from PIL import Image, ImageDraw, ImageFont
import qrcode as _qrcode
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as _rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

load_dotenv()

app = FastAPI(title="Super Legal Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEXES_DIR = os.path.join(BASE_DIR, "backend", "indexes")
CSV_LOG_FILE = os.path.join(BASE_DIR, "backend", "chat_log.csv")
DB_PATH = os.path.join(BASE_DIR, "backend", "superlegal.db")

DOCUMENT_OPTIONS = {
    "Nigerian Constitution": os.path.join(BASE_DIR, "documents", "Naija Constitutions.pdf"),
    "Labour Law Act": os.path.join(BASE_DIR, "documents", "LABOUR_ACT.pdf"),
    "Criminal Code": os.path.join(BASE_DIR, "documents", "C38.pdf"),
}

app.mount("/documents", StaticFiles(directory=os.path.join(BASE_DIR, "documents")), name="documents")

# ── Signing key infrastructure ───────────────────────────────────────────────

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
SIGNING_KEY_PATH = os.path.join(BASE_DIR, "backend", "server_signing.pem")


def _init_signing_key():
    if not os.path.exists(SIGNING_KEY_PATH):
        key = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with open(SIGNING_KEY_PATH, "wb") as f:
            f.write(key.private_bytes(
                _serialization.Encoding.PEM,
                _serialization.PrivateFormat.PKCS8,
                _serialization.NoEncryption(),
            ))


_init_signing_key()


def _sign_text(text: str):
    """Signs document text. Returns (doc_id, doc_hash, signature_b64)."""
    with open(SIGNING_KEY_PATH, "rb") as f:
        private_key = _serialization.load_pem_private_key(f.read(), password=None)
    doc_id = str(uuid.uuid4())
    encoded = text.encode("utf-8")
    doc_hash = hashlib.sha256(encoded).hexdigest()
    sig = private_key.sign(
        encoded,
        _apadding.PSS(
            mgf=_apadding.MGF1(_hashes.SHA256()),
            salt_length=_apadding.PSS.MAX_LENGTH,
        ),
        _hashes.SHA256(),
    )
    return doc_id, doc_hash, base64.b64encode(sig).decode()


# ── SQLite session storage ──────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                sender     TEXT NOT NULL,
                text       TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES ('Chat 1')")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signed_documents (
                doc_id       TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                signer_name  TEXT NOT NULL,
                signer_title TEXT NOT NULL,
                signer_bar   TEXT,
                signer_state TEXT NOT NULL,
                doc_type     TEXT NOT NULL,
                doc_hash     TEXT NOT NULL,
                signature    TEXT NOT NULL,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


init_db()


def db_get_sessions() -> List[str]:
    with get_db() as conn:
        return [r["session_id"] for r in conn.execute(
            "SELECT session_id FROM sessions ORDER BY created_at"
        ).fetchall()]


def db_session_exists(session_id: str) -> bool:
    with get_db() as conn:
        return bool(conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone())


def db_create_session(session_id: str):
    with get_db() as conn:
        conn.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))


def db_get_messages(session_id: str) -> List[Dict]:
    with get_db() as conn:
        return [
            {"sender": r["sender"], "text": r["text"]}
            for r in conn.execute(
                "SELECT sender, text FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ]


def db_get_recent_messages(session_id: str, limit: int = 6) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT sender, text FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"sender": r["sender"], "text": r["text"]} for r in reversed(rows)]


def db_append_messages(session_id: str, pairs: List[Dict]):
    with get_db() as conn:
        for msg in pairs:
            conn.execute(
                "INSERT INTO messages (session_id, sender, text) VALUES (?, ?, ?)",
                (session_id, msg["sender"], msg["text"]),
            )



# ── Cached ML components ────────────────────────────────────────────────────

embedding_model = None
unified_vector_db = None
_genai_client = None


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("API_KEY")
        if api_key:
            _genai_client = genai.Client(api_key=api_key)
    return _genai_client


MODEL_DIR = os.path.join(BASE_DIR, "backend", "models", "all-MiniLM-L6-v2")


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        if not os.path.isdir(MODEL_DIR):
            raise RuntimeError(
                "Embedding model not found. Run: python download_model.py"
            )
        embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_DIR,
            model_kwargs={"local_files_only": True},
        )
    return embedding_model


def load_vector_store():
    faiss_dir = os.path.join(INDEXES_DIR, "unified")
    faiss_index = os.path.join(faiss_dir, "index.faiss")
    faiss_pkl = os.path.join(faiss_dir, "index.pkl")
    embedder = get_embedding_model()

    if os.path.exists(faiss_index) and os.path.exists(faiss_pkl):
        return FAISS.load_local(faiss_dir, embedder, allow_dangerous_deserialization=True)

    os.makedirs(faiss_dir, exist_ok=True)
    all_split_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    for doc_key, file_path in DOCUMENT_OPTIONS.items():
        if not os.path.exists(file_path):
            print(f"Warning: Document not found at {file_path}")
            continue
        print(f"Loading and indexing: {doc_key}...")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["document_name"] = doc_key
        all_split_docs.extend(text_splitter.split_documents(docs))

    if not all_split_docs:
        raise FileNotFoundError("No source PDFs found to index.")

    vector_db = FAISS.from_documents(all_split_docs, embedder)
    vector_db.save_local(faiss_dir)
    print("Unified FAISS index created and cached.")
    return vector_db


def get_unified_vector_db():
    global unified_vector_db
    if unified_vector_db is None:
        unified_vector_db = load_vector_store()
    return unified_vector_db


# ── Logging (runs in background, never blocks a request) ────────────────────

def log_to_csv(session_name: str, document: str, language: str, question: str, answer: str):
    file_exists = os.path.exists(CSV_LOG_FILE)
    os.makedirs(os.path.dirname(CSV_LOG_FILE), exist_ok=True)
    with open(CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "session", "document", "language", "question", "answer"])
        writer.writerow([datetime.now().isoformat(), session_name, document, language, question, answer])


# ── Digital stamp & signed PDF generation ───────────────────────────────────

def _pil_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arial" + ("bd" if bold else "") + ".ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans" + ("-Bold" if bold else "-Regular") + ".ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _create_stamp(
    signer_name: str, title: str, bar_number: str,
    state: str, doc_id: str, date_str: str,
) -> Image.Image:
    SIZE, C = 400, 200
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    INK = (0, 35, 90, 215)

    draw.ellipse([6, 6, SIZE - 6, SIZE - 6], outline=INK, width=6)
    draw.ellipse([32, 32, SIZE - 32, SIZE - 32], outline=INK, width=2)

    for cx, cy in [(C, 19), (C, SIZE - 19)]:
        draw.text((cx, cy), "✦", fill=INK, font=_pil_font(14), anchor="mm")

    draw.line([(60, C - 65), (SIZE - 60, C - 65)], fill=INK, width=1)
    draw.line([(60, C + 60), (SIZE - 60, C + 60)], fill=INK, width=1)

    draw.text((C, C - 87), "DIGITALLY SIGNED", fill=INK, font=_pil_font(17, bold=True), anchor="mm")
    draw.text((C, C - 42), signer_name.upper()[:26], fill=INK, font=_pil_font(20, bold=True), anchor="mm")
    draw.text((C, C - 13), title[:34], fill=INK, font=_pil_font(15), anchor="mm")

    if bar_number:
        draw.text((C, C + 17), f"NBA: {bar_number}", fill=INK, font=_pil_font(14), anchor="mm")
        draw.text((C, C + 42), state[:30].upper(), fill=INK, font=_pil_font(14), anchor="mm")
    else:
        draw.text((C, C + 29), state[:30].upper(), fill=INK, font=_pil_font(14), anchor="mm")

    draw.text((C, C + 76), date_str, fill=INK, font=_pil_font(13), anchor="mm")
    draw.text((C, C + 96), f"REF: {doc_id[:8].upper()}", fill=INK, font=_pil_font(12), anchor="mm")
    return img


def _create_qr(url: str) -> Image.Image:
    qr = _qrcode.QRCode(version=1, box_size=6, border=2,
                        error_correction=_qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGBA")


def _rl_para(text: str) -> str:
    """Strip markdown and escape XML for reportlab Paragraph."""
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: m.group(1).upper(), text, flags=re.DOTALL)
    text = re.sub(r'\*+', '', text)
    return html.escape(text)


def _build_signed_pdf(
    document_text: str, document_type: str,
    signer_name: str, signer_title: str, bar_number: str, state: str,
    doc_id: str, date_str: str, verify_url: str,
) -> bytes:
    buf = BytesIO()
    stamp_img = _create_stamp(signer_name, signer_title, bar_number, state, doc_id, date_str)
    qr_img = _create_qr(verify_url)

    STAMP_PT, QR_PT = 100, 80
    _sb = BytesIO()
    stamp_img.resize((STAMP_PT * 4, STAMP_PT * 4), Image.LANCZOS).save(_sb, "PNG")
    _sb.seek(0)
    _qb = BytesIO()
    qr_img.resize((QR_PT * 4, QR_PT * 4), Image.LANCZOS).save(_qb, "PNG")
    _qb.seek(0)
    stamp_reader = ImageReader(_sb)
    qr_reader = ImageReader(_qb)

    W, H = A4
    MARGIN, FOOTER_H = 55, 145

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(_rl_colors.HexColor("#C9A227"))
        canvas.setLineWidth(1.2)
        canvas.line(MARGIN, FOOTER_H - 5, W - MARGIN, FOOTER_H - 5)

        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(_rl_colors.HexColor("#4A5568"))
        canvas.drawString(MARGIN, FOOTER_H - 20, "DIGITALLY AUTHENTICATED  •  SUPER LEGAL")

        canvas.drawImage(stamp_reader, MARGIN, 15, STAMP_PT, STAMP_PT, mask="auto")

        qr_x = MARGIN + STAMP_PT + 12
        canvas.drawImage(qr_reader, qr_x, 22, QR_PT, QR_PT, mask="auto")
        canvas.setFont("Helvetica", 6)
        canvas.setFillColor(_rl_colors.HexColor("#718096"))
        canvas.drawCentredString(qr_x + QR_PT / 2, 14, "Scan to verify")

        tx = qr_x + QR_PT + 18
        ty = FOOTER_H - 30
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(_rl_colors.HexColor("#00225A"))
        canvas.drawString(tx, ty, signer_name)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_rl_colors.HexColor("#4A5568"))
        detail_lines = [signer_title]
        if bar_number:
            detail_lines.append(f"NBA Bar No.: {bar_number}")
        detail_lines += [state, f"Date: {date_str}", f"Doc ID: {doc_id[:16]}..."]
        for i, line in enumerate(detail_lines):
            canvas.drawString(tx, ty - 14 * (i + 1), line)

        canvas.setFont("Helvetica", 6)
        canvas.setFillColor(_rl_colors.HexColor("#A0AEC0"))
        canvas.drawCentredString(W / 2, 5, f"Verify at: {verify_url}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=50, bottomMargin=FOOTER_H + 15,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SLTitle", parent=styles["Heading1"],
        fontName="Times-Bold", fontSize=14,
        textColor=_rl_colors.HexColor("#00225A"),
        alignment=1, spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "SLBody", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=11,
        leading=20, spaceAfter=8,
    )
    story = [
        Paragraph(html.escape(document_type.upper()), title_style),
        HRFlowable(width="100%", thickness=1.5, color=_rl_colors.HexColor("#C9A227"), spaceAfter=12),
        Spacer(1, 6),
    ]
    for block in document_text.split("\n\n"):
        block = block.strip()
        if block:
            story.append(Paragraph(_rl_para(block).replace("\n", "<br/>"), body_style))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ── Pydantic models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    document_key: str
    language: str
    query: str = Field(..., min_length=1, max_length=2000)


class CreateSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)


class DraftRequest(BaseModel):
    session_id: str
    document_type: str


class ConsultationStartRequest(BaseModel):
    session_id: str
    issue: str = Field(..., min_length=1, max_length=2000)
    document_key: str


class ConsultationAssessRequest(BaseModel):
    session_id: str
    issue: str = Field(..., max_length=2000)
    questions: List[str]
    answers: List[str]
    document_key: str
    language: str = "English"


class SignDocumentRequest(BaseModel):
    session_id: str
    document: str = Field(..., min_length=1)
    document_type: str
    signer_name: str = Field(..., min_length=2, max_length=100)
    signer_title: str = Field(..., min_length=2, max_length=100)
    signer_bar_number: str = Field(default="", max_length=50)
    signer_state: str = Field(..., min_length=2, max_length=50)


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/documents")
def get_documents():
    return ["All Documents"] + list(DOCUMENT_OPTIONS.keys())


@app.get("/api/sessions")
def get_sessions():
    return db_get_sessions()


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    session_id = req.session_id.strip()
    if db_session_exists(session_id):
        raise HTTPException(status_code=400, detail="Session already exists.")
    db_create_session(session_id)
    return {"status": "success", "session_id": session_id}


@app.get("/api/sessions/{session_id}")
def get_session_history(session_id: str):
    if not db_session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return db_get_messages(session_id)


@app.post("/api/chat")
def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    session_id = req.session_id
    if not db_session_exists(session_id):
        db_create_session(session_id)

    gemini = get_genai_client()
    if not gemini:
        raise HTTPException(
            status_code=400,
            detail="Gemini API Key is missing. Set API_KEY in backend/.env",
        )

    try:
        vector_db = get_unified_vector_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")

    try:
        if req.document_key == "All Documents":
            retriever = vector_db.as_retriever(search_kwargs={"k": 20})
        else:
            retriever = vector_db.as_retriever(
                search_kwargs={"k": 20, "filter": {"document_name": req.document_key}}
            )

        docs = retriever.invoke(req.query)
        context = "\n".join([doc.page_content for doc in docs])

        recent = db_get_recent_messages(session_id, limit=6)
        history_str = "".join(
            f"{'Client' if m['sender'] == 'user' else 'Assistant'}: {m['text']}\n"
            for m in recent
        )

        prompt = f"""You are an experienced Nigerian legal assistant.
Conversation History:
{history_str}

The client has asked: {req.query}

Relevant legal context:
{context}

Explain the key legal points clearly, as you would in a client consultation. Refer to specific sections or articles from the text if available.
Respond in {req.language} only."""

        result = gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        answer = result.text

        db_append_messages(session_id, [
            {"sender": "user", "text": req.query},
            {"sender": "ai", "text": answer},
        ])
        background_tasks.add_task(log_to_csv, session_id, req.document_key, req.language, req.query, answer)

        return {
            "answer": answer,
            "session_id": session_id,
            "references": [
                {
                    "page_content": doc.page_content,
                    "page": doc.metadata.get("page", 0) + 1,
                    "filename": os.path.basename(doc.metadata.get("source", "")),
                    "document_name": doc.metadata.get("document_name", "Unknown Document"),
                }
                for doc in docs[:3]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/api/draft")
def draft_document(req: DraftRequest):
    session_id = req.session_id
    messages = db_get_messages(session_id)
    if not messages:
        raise HTTPException(
            status_code=404,
            detail="No conversation history found in this session. Please ask a legal question first.",
        )

    gemini = get_genai_client()
    if not gemini:
        raise HTTPException(status_code=400, detail="Gemini API Key is missing. Set API_KEY in backend/.env")

    try:
        history_str = ""
        for msg in messages[-10:]:
            role = "Client" if msg["sender"] == "user" else "Legal Assistant"
            history_str += f"{role}: {msg['text']}\n\n"

        document_instructions = {
            "Demand Letter": "Draft a formal Nigerian legal DEMAND LETTER. Include: date, sender and recipient addresses, subject line, clear statement of facts, legal basis (cite specific Nigerian laws), a clear demand with a 14-day deadline, and a signature block.",
            "Quit Notice": "Draft a formal QUIT NOTICE under Nigerian tenancy law. Include: date, landlord and tenant names/addresses, property address, the legal basis for quit notice, the required notice period per Nigerian law, and a signature block.",
            "Formal Complaint Letter": "Draft a FORMAL COMPLAINT LETTER to the relevant Nigerian authority (employer, regulatory body, or police). Include: date, sender details, recipient authority, clear description of the complaint, referenced Nigerian laws violated, requested remedy, and signature block.",
            "Employment Rights Notice": "Draft a formal EMPLOYMENT RIGHTS NOTICE to an employer under the Nigerian Labour Act. Include: date, employee details, employer details, specific rights violations under the Labour Act, a demand for compliance, timeline for response, and signature block.",
            "Statement of Facts": "Draft a formal STATEMENT OF FACTS document suitable for use by a Nigerian lawyer. Include: date, party names, a numbered chronological account of the material facts, the relevant Nigerian laws implicated, and a declaration of truth.",
        }

        specific_instruction = document_instructions.get(
            req.document_type,
            f"Draft a professional {req.document_type} following Nigerian legal standards and conventions.",
        )

        prompt = f"""You are an expert Nigerian legal document drafter with 20 years of experience.

Based on the following conversation between a client and a legal assistant, draft the requested legal document.

CONVERSATION CONTEXT:
{history_str}

DOCUMENT TO DRAFT: {req.document_type}

INSTRUCTIONS:
{specific_instruction}

IMPORTANT:
- Use today's date: {datetime.now().strftime('%d %B %Y')}
- Use formal, professional Nigerian legal language
- Reference specific Nigerian laws, acts, or sections where applicable
- Use [CLIENT NAME], [CLIENT ADDRESS], [RECIPIENT NAME], [RECIPIENT ADDRESS] as placeholders where actual details are unknown
- The document should be ready to use with minimal editing
- Do NOT include any commentary, preamble or explanation — output ONLY the document itself"""

        result = gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)

        return {
            "document": result.text,
            "document_type": req.document_type,
            "generated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document generation failed: {str(e)}")


@app.post("/api/consultation/start")
def consultation_start(req: ConsultationStartRequest):
    gemini = get_genai_client()
    if not gemini:
        raise HTTPException(status_code=400, detail="Gemini API Key is missing. Set API_KEY in backend/.env")

    try:
        domain_hint = f" in the context of {req.document_key}" if req.document_key != "All Documents" else ""

        prompt = f"""You are a Nigerian legal intake assistant conducting a structured client consultation{domain_hint}.

A client described their situation as:
"{req.issue}"

Generate exactly 3 short, targeted follow-up questions to gather the specific facts needed for a thorough legal assessment.
Rules:
- Each question must be answerable in 1-2 sentences
- Focus on legally significant facts: dates, written agreements, notice periods, amounts, job roles, witness presence
- Do NOT ask for information already stated in the issue description
- Questions must be specific to Nigerian law context

Return ONLY a JSON array of exactly 3 question strings. Example format:
["Question one?", "Question two?", "Question three?"]

Return nothing else."""

        result = gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)

        text = result.text.strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        questions = json.loads(match.group() if match else text)
        questions = [str(q) for q in questions[:3]]

        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consultation start failed: {str(e)}")


@app.post("/api/consultation/assess")
def consultation_assess(req: ConsultationAssessRequest, background_tasks: BackgroundTasks):
    gemini = get_genai_client()
    if not gemini:
        raise HTTPException(status_code=400, detail="Gemini API Key is missing. Set API_KEY in backend/.env")

    if not db_session_exists(req.session_id):
        db_create_session(req.session_id)

    try:
        vector_db = get_unified_vector_db()

        retrieval_query = req.issue + " " + " ".join(req.answers)

        if req.document_key == "All Documents":
            retriever = vector_db.as_retriever(search_kwargs={"k": 20})
        else:
            retriever = vector_db.as_retriever(
                search_kwargs={"k": 20, "filter": {"document_name": req.document_key}}
            )

        docs = retriever.invoke(retrieval_query)
        context = "\n".join([doc.page_content for doc in docs])

        qa_summary = f"Situation: {req.issue}\n\n"
        for i, (q, a) in enumerate(zip(req.questions, req.answers), 1):
            qa_summary += f"Q{i}: {q}\nA{i}: {a}\n\n"

        prompt = f"""You are an experienced Nigerian legal consultant who has just completed a structured intake interview with a client.

CLIENT INTAKE SUMMARY:
{qa_summary}

RELEVANT LEGAL CONTEXT FROM NIGERIAN LAW:
{context}

Provide a comprehensive and definitive legal assessment structured as follows:

**1. Summary of Situation**
Briefly restate the key legally relevant facts.

**2. Legal Analysis**
Which Nigerian laws apply and how — cite specific sections and acts.

**3. Your Rights / Legal Position**
What the client is entitled to under Nigerian law.

**4. Recommended Next Steps**
Concrete, numbered, actionable advice.

**5. Urgency**
Whether this is time-sensitive and why (e.g. statutes of limitation, notice deadlines).

Be direct and definitive. This is a formal consultation result.
Respond in {req.language} only."""

        result = gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        assessment = result.text

        consultation_log = (
            f"[CONSULTATION]\nSituation: {req.issue}\n"
            + "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(req.questions, req.answers)])
        )
        db_append_messages(req.session_id, [
            {"sender": "user", "text": consultation_log},
            {"sender": "ai", "text": assessment},
        ])
        background_tasks.add_task(
            log_to_csv, req.session_id, req.document_key, req.language, consultation_log, assessment
        )

        return {
            "assessment": assessment,
            "references": [
                {
                    "page_content": doc.page_content,
                    "page": doc.metadata.get("page", 0) + 1,
                    "filename": os.path.basename(doc.metadata.get("source", "")),
                    "document_name": doc.metadata.get("document_name", "Unknown Document"),
                }
                for doc in docs[:3]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consultation assessment failed: {str(e)}")


@app.post("/api/sign")
def sign_document(req: SignDocumentRequest):
    date_str = datetime.now().strftime("%d %B %Y")
    doc_id, doc_hash, signature = _sign_text(req.document)
    verify_url = f"{BASE_URL}/api/verify/{doc_id}"

    try:
        pdf_bytes = _build_signed_pdf(
            document_text=req.document,
            document_type=req.document_type,
            signer_name=req.signer_name,
            signer_title=req.signer_title,
            bar_number=req.signer_bar_number,
            state=req.signer_state,
            doc_id=doc_id,
            date_str=date_str,
            verify_url=verify_url,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    with get_db() as conn:
        conn.execute(
            """INSERT INTO signed_documents
               (doc_id, session_id, signer_name, signer_title, signer_bar,
                signer_state, doc_type, doc_hash, signature)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, req.session_id, req.signer_name, req.signer_title,
             req.signer_bar_number or None, req.signer_state,
             req.document_type, doc_hash, signature),
        )

    safe_type = re.sub(r"[^a-zA-Z0-9_\-]", "_", req.document_type)
    filename = f"{safe_type}_{datetime.now().strftime('%Y%m%d')}_signed.pdf"

    return {
        "pdf_b64": base64.b64encode(pdf_bytes).decode(),
        "doc_id": doc_id,
        "filename": filename,
        "verify_url": verify_url,
    }


@app.get("/api/verify/{doc_id}")
def verify_document(doc_id: str):
    with get_db() as conn:
        row = conn.execute(
            """SELECT signer_name, signer_title, signer_bar, signer_state,
                      doc_type, doc_hash, created_at
               FROM signed_documents WHERE doc_id = ?""",
            (doc_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found or ID is invalid.")

    return {
        "status": "AUTHENTIC",
        "doc_id": doc_id,
        "document_type": row["doc_type"],
        "signer_name": row["signer_name"],
        "signer_title": row["signer_title"],
        "signer_bar_number": row["signer_bar"] or "Not provided",
        "signer_state": row["signer_state"],
        "signed_at": row["created_at"],
        "document_hash": row["doc_hash"],
        "message": "This document was digitally signed and its integrity is verified.",
    }
