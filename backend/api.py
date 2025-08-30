import os
import uuid
import time
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.config import UPLOAD_DIR
from backend.rag_pipeline import add_documents, get_qa_chain, reset_scope

load_dotenv()

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

def is_mime_allowed(ext: str, mime_type: str) -> bool:
    if not mime_type:
        return True
    mime = mime_type.lower()
    if ext == ".pdf":
        return "pdf" in mime
    if ext == ".txt":
        return mime.startswith("text")
    if ext == ".docx":
        return ("word" in mime) or ("officedocument" in mime) or ("openxmlformats" in mime)
    return False

app = FastAPI(title="Mini RAG (Embedded Qdrant + Reranker + Groq)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Mini RAG backend running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), scope: str = Query("default"), fresh: bool = Query(False)):
    """
    Upload .txt/.pdf/.docx and index into embedded Qdrant under 'scope'.
    Set fresh=true to clear previous vectors in this scope.
    """
    ext = os.path.splitext(file.filename or "")[-1].lower()
    mime_type = (file.content_type or "").lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .txt, .pdf, and .docx files are supported.")
    if not is_mime_allowed(ext, mime_type):
        raise HTTPException(status_code=400, detail=f"Invalid MIME {mime_type!r} for extension {ext}.")

    base_name = os.path.splitext(os.path.basename(file.filename or "upload"))
    safe_filename = f"{base_name}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await file.close()

    if fresh:
        reset_scope(scope)

    return add_documents(file_path, scope=scope)  # [1][6]

@app.post("/upload_text")
async def upload_text(body: Dict[str, Any]):
    """
    Index pasted text directly; accepts { "text": "...", "scope": "..." , "fresh": bool }.
    """
    text = (body or {}).get("text", "")
    scope = (body or {}).get("scope", "default")
    fresh = bool((body or {}).get("fresh", False))

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")

    if fresh:
        reset_scope(scope)

    filename = f"pasted_{uuid.uuid4().hex[:8]}.txt"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return add_documents(path, scope=scope)

@app.post("/query")
async def query_rag(body: Dict[str, Any]):
    """
    Query RAG with reranking; accepts { "question": "...", "scope": "..." }.
    Returns answer, inline citations mapping, and rough timing.
    """
    question = (body or {}).get("question")
    scope = (body or {}).get("scope", "default")

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        qa = get_qa_chain(scope=scope)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    t0 = time.time()
    try:
        out = qa.invoke({"query": question})
    except Exception as e:
        msg = str(e)
        if "decommissioned" in msg or ("model" in msg and "supported" in msg):
            raise HTTPException(status_code=400, detail="Groq model deprecated/unsupported. Update GROQ_MODEL_ID.")
        if "Invalid API Key" in msg or "401" in msg:
            raise HTTPException(status_code=400, detail="Groq authentication failed: check GROQ_API_KEY.")
        raise

    latency_ms = int((time.time() - t0) * 1000)

    docs = out.get("source_documents", []) or []
    citations = []
    for idx, d in enumerate(docs, start=1):
        snippet = (d.page_content or "")[:400]
        meta = d.metadata or {}
        citations.append({
            "marker": f"[{idx}]",
            "source": meta.get("source", "Unknown"),
            "section": meta.get("section"),
            "chunk_id": meta.get("chunk_id"),
            "position": meta.get("position"),
            "snippet": snippet
        })

    return {
        "answer": out.get("result", ""),
        "citations": citations,
        "metrics": {"latency_ms": latency_ms}
    }
