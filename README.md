#  Super Legal – Multi-Document Legal RAG Chat App

Super Legal is an AI-powered legal assistant built with **Streamlit**, **LangChain**, and **Google Gemini**.  
It allows users to ask questions about different Nigerian laws (e.g., the Constitution, Labour Law Act) and get answers in natural language.  
The system uses **Retrieval-Augmented Generation (RAG)** with **FAISS** for semantic search over legal documents.

---

##  Features

-  **Multi-Document Support** – Switch between Nigerian Constitution, Labour Law Act, etc.  
-  **Chat-Like Interface** – Conversations look and feel like WhatsApp chats.  
-  **Multilingual Responses** – Get answers in English, French, Yoruba, Hausa, or Igbo.  
-  **Chat History** – Each chat session saves your previous Q&A for easy reference.  
-  **Semantic Search (FAISS)** – Efficient retrieval of relevant sections before answering.  
-  **AI-Powered Answers** – Uses Google Gemini (via `google-generativeai`) to generate clear responses.  
-  **Data Collection for Fine-Tuning** – Automatically logs questions and answers for future model fine-tuning (e.g., FLAN-T5).  

---

##  Tech Stack

- [Streamlit](https://streamlit.io/) – Web UI  
- [LangChain](https://www.langchain.com/) – RAG pipeline  
- [FAISS](https://faiss.ai/) – Vector search  
- [SentenceTransformers](https://www.sbert.net/) – Embeddings (`all-MiniLM-L6-v2`)  
- [Google Generative AI](https://ai.google.dev/) – LLM for answers  
- [dotenv](https://pypi.org/project/python-dotenv/) – Environment management  

---

Usage

Select a document (e.g., Constitution, Labour Act).

Type your legal question in the chat box.

Choose your preferred language (English, French, Yoruba, Hausa, Igbo).

Get AI-powered answers with references from the legal text.

Start new chats anytime, previous conversations remain accessible.

Disclaimer

This project is for educational and research purposes only.
It is not a substitute for professional legal advice.
Always consult a licensed lawyer for actual legal matters.

Contributing

Fork the repo

Create a new branch (feature-xyz)

Commit changes

Open a Pull Request

