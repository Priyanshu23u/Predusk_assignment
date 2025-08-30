import streamlit as st
import requests
import time
import streamlit.components.v1 as components

# ----------------------------
# Auto Scroll Function
# ----------------------------
def scroll_to_bottom():
    """JavaScript to scroll to bottom of page"""
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
# Custom Styling
# ----------------------------
THEME_CSS = """
<style>
/* Page background */
html, body, [data-testid="stAppViewContainer"] {
  background-color: #0f172a;
  color: #e2e8f0;
}

/* Reduce vertical spacing */
.block-container {
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
  max-width: 100% !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background-color: #111827;
  border-right: 1px solid #1f2937;
}

/* Chat message bubbles */
.user-msg {
  background-color: #0369a1;
  color: #f0f9ff;
  padding: 0.75rem 1rem;
  border-radius: 18px 18px 5px 18px;
  margin: 0.25rem 0 0.5rem auto;
  max-width: 80%;
  width: fit-content;
  word-wrap: break-word;
}

.assistant-msg {
  background-color: #374151;
  color: #f3f4f6;
  padding: 0.75rem 1rem;
  border-radius: 18px 18px 18px 5px;
  margin: 0.25rem auto 0.5rem 0;
  max-width: 80%;
  width: fit-content;
  word-wrap: break-word;
}

/* Input styling */
.stTextInput > div > div > input {
  background-color: #111827 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
  border-radius: 20px !important;
  padding: 0.75rem 1rem !important;
}

.stTextArea textarea {
  background-color: #111827 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
  min-height: 80px !important;
}

/* Buttons */
.stButton > button {
  background-color: #1f2937 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
  border-radius: 20px !important;
  height: 3rem !important;
}
.stButton > button:hover {
  border-color: #60a5fa !important;
  background-color: #374151 !important;
}

/* File uploader */
.stFileUploader div[data-testid="stFileUploaderDropzone"] {
  background-color: #111827 !important;
  border: 2px dashed #334155 !important;
  border-radius: 10px !important;
}

/* Remove excessive padding */
div[data-testid="stVerticalBlock"] > div {
  gap: 0.25rem !important;
}

/* Hide Streamlit branding */
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------
# Backend Config & Helper
# ----------------------------
BACKEND_URL = "http://localhost:8000"

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

# Display chat messages
if not st.session_state.messages:
    st.markdown(
        '<div class="assistant-msg">👋 Hello! Upload some documents and ask me questions about them.</div>',
        unsafe_allow_html=True
    )

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'<div class="user-msg">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg">{content}</div>', unsafe_allow_html=True)
        
        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📄 Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{source.get('marker', f'[{i}]')} {source.get('source', 'Unknown')}**")
                    if source.get('section'):
                        st.text(f"Section: {source.get('section')}")
                    st.code(source.get('snippet', ''), language='text')

# Input area
st.markdown("---")

# Use form to prevent automatic rerun on text change
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Message", 
            placeholder="Ask something about your documents...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.form_submit_button("➤", use_container_width=True)

# Handle message sending - only process if not already processing
if send_button and user_input.strip() and not st.session_state.processing:
    # Set processing flag to prevent duplicate submissions
    st.session_state.processing = True
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get response from backend
    with st.spinner("🤔 Thinking..."):
        t0 = time.time()
        res = requests.post(f"{BACKEND_URL}/query", json={"question": user_input, "scope": scope})
        data = _safe_json(res)
        elapsed = int((time.time() - t0) * 1000)
    
    if res.ok and data:
        answer = data.get("answer", "")
        sources = data.get("citations", [])
        
        # Add assistant message
        assistant_message = {
            "role": "assistant", 
            "content": f"{answer}\n\n⏱️ {elapsed}ms",
            "sources": sources
        }
        st.session_state.messages.append(assistant_message)
    else:
        error_msg = f"❌ Error: {(data or {}).get('detail', res.text)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Reset processing flag
    st.session_state.processing = False
    
    # Scroll to bottom and rerun to show new messages
    scroll_to_bottom()
    st.rerun()

# Footer
st.markdown(f"🏷️ **Scope:** {scope}")
