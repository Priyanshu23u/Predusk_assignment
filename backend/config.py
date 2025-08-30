import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant embedded (no Docker / no external server)
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "data/qdrant_local")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mini_rag")
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "cosine")  # cosine | dot | euclid

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # MiniLM-L6-v2 => 384

# Chunking strategy
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # ~12%

# Retrieval + Reranker
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "12"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# LLM (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

# Uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
