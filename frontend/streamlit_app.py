import os
import time
import requests
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------
# Auto Scroll Function
# ----------------------------
def scroll_to_bottom():
    components.html(
        """
        <script>
        setTimeout(function() {
            window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
        }, 100);
        </script>
        """,
        height=0,
    )

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(page_title="Mini RAG Chat", layout="wide")

# ----------------------------
# Custom Styling (compact, dark)
# ----------------------------
THEME_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #0f172a; color: #e2e8f0; }
[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
.block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
.user-msg { background-color: #0369a1; color: #f0f9ff; padding: 0.75rem 1rem; border-radius: 18px 18px 5px 18px; margin: 0.25rem 0 0.5rem auto; max-width: 80%; width: fit-content; }
.assistant-msg { background-color: #374151; color: #f3f4f6; padding: 0.75rem 1rem; border-radius: 18px 18px 18px 5px; margin: 0.25rem auto 0.5rem 0; max-width: 80%; width: fit-content; }
.stTextInput > div > div > input, .stTextArea textarea { background-color: #111827 !important; color: #e5e7eb !important; border: 1px solid #334155 !important; border-radius: 10px !important; }
.stButton > button { background-color: #1f2937 !important; color: #e5e7eb !important; border: 1px solid #334155 !important; border-radius: 20px !important; height: 3rem !important; }
.stButton > button:hover { border-color: #60a5fa !important; background-color: #374151 !important; }
.stFileUploader div[data-testid="stFileUploaderDropzone"] { background-color: #111827 !important; border: 2px dashed #334155 !important; border-radius: 10px !important; }
.source-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0; }
div[data-testid="stVerticalBlock"] > div { gap: 0.25rem !important; }
footer, header { visibility: hidden; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------
# Backend Config & Helper
# ----------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def _safe_json(res):
    try:
        return res.json()
    except Exception:
        return None

# ----------------------------
# Initialize Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.markdown("### 🔧 Settings")
    scope = st.text_input("Session Scope", value="default")
    fresh = st.checkbox("Fresh scope on upload", value=False)
    st.markdown("---")

    # Upload section
    st.markdown("### 📁 Upload Content")
    uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf", "docx"])
    if uploaded_file and st.button("🔄 Index File", use_container_width=True):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
        params = {"scope": scope, "fresh": str(fresh).lower()}
        with st.spinner("Indexing..."):
            t0 = time.time()
            res = requests.post(f"{BACKEND_URL}/upload", files=files, params=params)
            data = _safe_json(res)
        if res.ok and data:
            st.success(f"✅ Indexed ({int((time.time()-t0)*1000)} ms)")
        else:
            st.error(f"❌ {(data or {}).get('detail', res.text)}")

    st.markdown("---")
    text = st.text_area("Paste text", height=80, placeholder="Paste content...")
    if st.button("🔄 Index Text", use_container_width=True):
        if text.strip():
            with st.spinner("Indexing..."):
                t0 = time.time()
                res = requests.post(f"{BACKEND_URL}/upload_text", json={"text": text, "scope": scope, "fresh": fresh})
                data = _safe_json(res)
            if res.ok and data:
                st.success(f"✅ Indexed ({int((time.time()-t0)*1000)} ms)")
            else:
                st.error(f"❌ {(data or {}).get('detail', res.text)}")

# ----------------------------
# Main Chat Interface
# ----------------------------
st.title("💬 Mini RAG Chat")
st.caption("Ask questions about your uploaded documents")

if not st.session_state.messages:
    st.markdown('<div class="assistant-msg">👋 Hello! Upload documents and ask questions about them.</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    if role == "user":
        st.markdown(f'<div class="user-msg">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg">{content}</div>', unsafe_allow_html=True)
        if "sources" in message and message["sources"]:
            with st.expander("📄 Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{source.get('marker', f'[{i}]')} {source.get('source', 'Unknown')}**")
                    if source.get('section'):
                        st.text(f"Section: {source.get('section')}")
                    st.code(source.get('snippet', ''), language='text')

st.markdown("---")

# Use a form to avoid rerun on every keystroke and prevent loops
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Message", placeholder="Ask something about your documents...", label_visibility="collapsed")
    with col2:
        send_button = st.form_submit_button("➤", use_container_width=True)

if send_button and user_input.strip() and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("🤔 Thinking..."):
        t0 = time.time()
        res = requests.post(f"{BACKEND_URL}/query", json={"question": user_input, "scope": scope})
        data = _safe_json(res)
        elapsed = int((time.time() - t0) * 1000)

    if res.ok and data:
        answer = data.get("answer", "")
        sources = data.get("citations", [])
        assistant_message = {
            "role": "assistant",
            "content": f"{answer}\n\n⏱️ {elapsed}ms",
            "sources": sources
        }
        st.session_state.messages.append(assistant_message)
    else:
        st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {(data or {}).get('detail', res.text)}"})

    st.session_state.processing = False
    scroll_to_bottom()
    st.rerun()

st.markdown(f"🏷️ **Scope:** {scope}")
