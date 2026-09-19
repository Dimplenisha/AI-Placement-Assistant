# 🤖 AI Placement Assistant

An AI-powered placement document assistant that allows students to upload
placement notifications and job descriptions and ask questions about them.

## 🚀 Features

- Upload PDF placement documents
- Extract text from PDFs
- Split documents into meaningful chunks
- Generate local text embeddings
- Perform semantic search
- Perform keyword-based search
- Hybrid document retrieval
- Gemini-powered question answering
- Retrieval-Augmented Generation (RAG)
- Multiple PDF support
- Source document identification
- Modern Streamlit web interface

## 🧠 Architecture

PDF Documents
       ↓
Text Extraction
       ↓
Text Chunking
       ↓
Sentence Embeddings
       ↓
Hybrid Retrieval
       ↓
Relevant Context
       ↓
Gemini LLM
       ↓
Grounded Answer

## 🛠️ Technologies

- Python
- Streamlit
- Gemini API
- Sentence Transformers
- scikit-learn
- PyPDF
- Retrieval-Augmented Generation (RAG)

## 📂 Project Structure

```text
AI-Placement-Assistant/
│
├── app.py
├── rag_engine.py
├── requirements.txt
├── README.md
├── documents/
└── uploaded_documents/
