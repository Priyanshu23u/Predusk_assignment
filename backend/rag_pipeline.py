import os
import uuid
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
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

# Embedded Qdrant (stores vectors under QDRANT_LOCAL_PATH)
qclient = QdrantClient(path=QDRANT_LOCAL_PATH)

def _ensure_collection():
    """
    Ensure local embedded Qdrant has the configured collection and vector params.
    """
    metric = {
        "cosine": qmodels.Distance.COSINE,
        "dot": qmodels.Distance.DOT,
        "euclid": qmodels.Distance.EUCLID
    }.get(QDRANT_DISTANCE.lower(), qmodels.Distance.COSINE)

    collections = qclient.get_collections().collections
    existing = [c.name for c in collections]
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

# Explicit chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
)

def _load_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", file_path)
    return docs

def _upsert_documents(docs, scope: str):
    """
    Split, annotate metadata, and upsert with valid UUID point IDs.
    The human-friendly chunk_id stays in metadata for scope filtering and citations.
    """
    chunks = splitter.split_documents(docs)
    ids: List[str] = []
    for i, d in enumerate(chunks):
        base = os.path.basename(d.metadata.get("source", "unknown"))
        chunk_id = f"{scope}:{base}:{i}"  # used in metadata (filtering/citations)
        d.metadata["chunk_id"] = chunk_id
        d.metadata.setdefault("section", d.metadata.get("page"))
        d.metadata["position"] = i
        # Qdrant embedded requires valid UUIDs or integers for point ids
        ids.append(str(uuid.uuid4()))
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)

def reset_scope(scope: str):
    """
    Delete all points for a given scope by filtering on the metadata.chunk_id prefix.
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
    )

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

    ce = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor = CrossEncoderReranker(model=ce, top_n=RERANK_TOP_N)

    return ContextualCompressionRetriever(
        base_retriever=base_retriever, base_compressor=compressor
    )

def get_qa_chain(scope: Optional[str] = "default"):
    """
    QA chain with Groq + embedded Qdrant retriever + cross-encoder reranker.
    """
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
