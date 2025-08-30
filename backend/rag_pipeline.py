import os
import uuid
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq

from backend.config import (
    QDRANT_LOCAL_PATH, QDRANT_COLLECTION, QDRANT_DISTANCE,
    EMBEDDING_MODEL, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP,
    RETRIEVE_K, RERANK_TOP_N, RERANKER_MODEL,
    GROQ_API_KEY, GROQ_MODEL_ID,
)

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Embedded Qdrant (local, no Docker needed)
qclient = QdrantClient(path=QDRANT_LOCAL_PATH)

def _ensure_collection():
    metric = {
        "cosine": qmodels.Distance.COSINE,
        "dot": qmodels.Distance.DOT,
        "euclid": qmodels.Distance.EUCLID
    }.get(QDRANT_DISTANCE.lower(), qmodels.Distance.COSINE)
    existing = [c.name for c in qclient.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        qclient.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBEDDING_DIM, distance=metric),
        )

_ensure_collection()

vectorstore = QdrantVectorStore(
    client=qclient,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

# Explicit chunking strategy
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
)

def _load_document(file_path: str):
    """
    Load a document by extension:
      - .txt via TextLoader (utf-8)
      - .pdf via PyPDFLoader (requires pypdf)
      - .docx via docx2txt to plain text
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)  # pypdf must be installed
        docs = loader.load()
    elif ext == ".docx":
        text = docx2txt.process(file_path)
        from langchain.schema import Document
        docs = [Document(page_content=text, metadata={"source": file_path})]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    for d in docs:
        d.metadata.setdefault("source", file_path)
    return docs  # [2][4]

def _upsert_documents(docs, scope: str):
    """
    Split, annotate metadata, and upsert with valid UUID point IDs.
    Keep the human-readable chunk_id in metadata for filtering and citations.
    """
    chunks = splitter.split_documents(docs)
    ids: List[str] = []
    for i, d in enumerate(chunks):
        base = os.path.basename(d.metadata.get("source", "unknown"))
        chunk_id = f"{scope}:{base}:{i}"
        d.metadata["chunk_id"] = chunk_id
        d.metadata.setdefault("section", d.metadata.get("page"))
        d.metadata["position"] = i
        ids.append(str(uuid.uuid4()))
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)

def reset_scope(scope: str):
    """
    Delete all points for a given scope by filtering on metadata.chunk_id prefix.
    Important: use 'metadata.chunk_id' with QdrantVectorStore payload layout.
    """
    prefix = f"{scope}:"
    qclient.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="metadata.chunk_id",
                    match=qmodels.MatchText(text=prefix)
                )]
            )
        )
    )  # [3][5]

def add_documents(file_path: str, scope: Optional[str] = "default") -> Dict[str, Any]:
    docs = _load_document(file_path)
    if not docs:
        raise ValueError("No text extracted from document.")
    added = _upsert_documents(docs, scope=scope or "default")
    return {"message": f"Indexed {added} chunks for {os.path.basename(file_path)} in scope '{scope}'."}

def get_retriever_with_reranker(scope: Optional[str] = "default"):
    """
    Retrieve top-k filtered by scope, then rerank to top_n via a local cross-encoder.
    """
    prefix = f"{scope}:"
    scope_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(
            key="metadata.chunk_id",
            match=qmodels.MatchText(text=prefix)
        )]
    )

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVE_K, "filter": scope_filter}
    )

    reranker_model = RERANKER_MODEL or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce = HuggingFaceCrossEncoder(model_name=reranker_model)
    compressor = CrossEncoderReranker(model=ce, top_n=RERANK_TOP_N)

    return ContextualCompressionRetriever(
        base_retriever=base_retriever, base_compressor=compressor
    )  # [3]

def get_qa_chain(scope: Optional[str] = "default"):
    retriever = get_retriever_with_reranker(scope=scope or "default")

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in .env")

    llm = ChatGroq(model=GROQ_MODEL_ID, groq_api_key=GROQ_API_KEY)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa
