# 📚 Mini RAG — Embedded Qdrant + Cross-Encoder Reranker

A modern **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **Qdrant**, **LangChain**, and **Groq** that provides intelligent document search and question-answering with source citations.

## 🎯 Features

- **📄 Multi-format Document Support**: Upload `.txt`, `.pdf`, and `.docx` files
- **💬 WhatsApp-style Chat Interface**: Clean, dark-themed conversational UI
- **🔍 Advanced Retrieval**: Vector similarity search with **reranking** using cross-encoder models
- **🎯 Source Citations**: Inline citations with document snippets and metadata
- **🚀 Free & Self-hosted**: No external dependencies - runs entirely locally
- **⚡ Fast Response Times**: Embedded Qdrant vector store with efficient retrieval
- **🔒 Session Isolation**: Scope-based document separation for multiple contexts

## 🏗️ Architecture
graph LR
A[Document Upload] --> B[Text Splitting]
B --> C[Embedding Generation]
C --> D[Qdrant Vector Store]
E[User Query] --> F[Vector Retrieval]
F --> G[Cross-Encoder Reranking]
G --> H[Context Assembly]
H --> I[Groq LLM]
I --> J[Answer + Citations]


## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API Key (free tier available)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Mini_RAG
```

2. **Install dependencies**
```bash
pip install -r backend/requirements.txt
```

3. **Start the backend**
```bash
uvicorn backend.api:app --reload
```

4. **Launch the frontend**
```bash 
streamlit run frontend/streamlit_app.py
```

6. **Access the app**
- Open your browser to `http://localhost:8501`
- Backend API docs available at `http://localhost:8000/docs`

## 📁 Project Structure
```bash 
mini-rag/
├── backend/
│ ├── init.py
│ ├── api.py # FastAPI endpoints
│ ├── rag_pipeline.py # RAG logic with reranking
│ ├── config.py # Configuration settings
│ ├── utils.py # Helper functions
│ └── requirements.txt
├── frontend/
│ ├── streamlit_app.py # Streamlit UI
│ ├── utils.py # Frontend helpers
│ └── requirements.txt
├── data/
│ ├── uploads/ # Uploaded files
│ └── qdrant_local/ # Embedded vector database
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```


## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | **Required**: Groq API key |
| `GROQ_MODEL_ID` | `llama-3.3-70b-versatile` | Groq model to use |
| `QDRANT_LOCAL_PATH` | `data/qdrant_local` | Embedded Qdrant storage path |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks (~12%) |
| `RETRIEVE_K` | `12` | Initial retrieval count |
| `RERANK_TOP_N` | `4` | Final reranked document count |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranking model |

### Chunking Strategy
- **Size**: 1,000 characters with 120-character overlap (~12%)
- **Method**: Recursive text splitting on paragraphs, sentences, then words
- **Metadata**: Preserves source file, section/page, position for citations

### Retrieval & Reranking
- **Stage 1**: Vector similarity search (top-k=12) using MiniLM-L6-v2 embeddings
- **Stage 2**: Cross-encoder reranking (top-n=4) using BAAI/bge-reranker-base
- **Distance Metric**: Cosine similarity in 384-dimensional space

## 💡 Usage

### 1. Document Management
- **Upload Files**: Use sidebar to upload documents (supports drag & drop)
- **Paste Content**: Direct text input for quick indexing
- **Session Scopes**: Isolate document sets using custom scope names
- **Fresh Sessions**: Clear previous documents when uploading new ones

### 2. Querying
- **Natural Language**: Ask questions in conversational style
- **Contextual**: References are maintained within scope sessions
- **Source Tracking**: Expandable citations show document snippets

### 3. Advanced Features
- **Multi-document Search**: Query across all uploaded documents in a scope
- **Scope Isolation**: Switch between different document collections
- **Real-time Metrics**: Response latency and processing times displayed


