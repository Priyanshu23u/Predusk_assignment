import requests

import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def upload_document(file_path):
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "application/octet-stream")}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)
    response.raise_for_status()
    return response.json()

def ask_question(query):
    response = requests.post(f"{BACKEND_URL}/query", json={"question": query})
    response.raise_for_status()
    return response.json()
