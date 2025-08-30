import os
import re
from typing import List

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max(1, chunk_size - overlap)
    return chunks
