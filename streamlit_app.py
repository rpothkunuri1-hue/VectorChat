# streamlit_app.py - Streamlit UI for RAG Application
"""
Streamlit frontend for RAG Application
Run with: streamlit run streamlit_app.py
Requires: pip install streamlit requests
"""

import streamlit as st
import requests
import json
from typing import List, Dict

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RAG Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-section {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def check_backend_connection():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_available_models():
    """Fetch available Ollama models from backend"""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", []), data.get("current", "llama3")
        return ["llama3", "mistral", "phi", "gemma"], "llama3"
    except:
        return ["llama3", "mistral", "phi", "gemma"], "llama3"

def get_system_health():
    """Get system health information"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def upload_document(uploaded_file):
    """Upload document to backend"""
    try:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        response = requests.post(f"{API_URL}/upload", files=files, timeout=30)
        return response
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def query_documents(question: str, model: str, top_k: int = 4):
    """Query the RAG system"""
    try:
        payload = {
            "question": question,
            "model": model,
            "top_k": top_k
        }
        response = requests.post(
            f"{API_URL}/query",
            json=payload,
            timeout=60
        )
        return response
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return None

def get_documents_list():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except:
        return []

def delete_document(filename: str):
    """Delete a document"""
    try:
        response = requests.delete(f"{API_URL}/documents/{filename}", timeout=5)
        return response.status_code == 200
    except:
        return False

def clear_all_documents():
    """Clear all documents and vector database"""
    try:
        response = requests.delete(f"{API_URL}/clear", timeout=10)
        return response.status_code == 200
    except:
        return False

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3"

# Main title
st.title("🤖 RAG Application - Document Q&A")
st.markdown("*Retrieval-Augmented Generation with Local LLMs*")

# Check backend connection
if not check_backend_connection():
    st.error("⚠️ Cannot connect to backend server. Please ensure the backend is running at http://localhost:8000")
    st.info("Start the backend with: `python rag_app.py`")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    available_models, current_model = get_available_models()
    selected_model = st.selectbox(
        "Select LLM Model",
        available_models,
        index=available_models.index(current_model) if current_model in available_models else 0,
        help="Choose the Ollama model for answer generation"
    )
    st.session_state.selected_model = selected_model
    
    # Retrieval settings
    st.subheader("Retrieval Settings")
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        help="More chunks provide more context but may be slower"
    )
    
    st.divider()
    
    # System health
    st.header("📊 System Status")
    health = get_system_health()
    
    if health:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "🟢 Online")
        with col2:
            st.metric("Vectors", health.get("documents_indexed", 0))
    else:
        st.metric("Status", "🔴 Offline")
    
    st.divider()
    
    # Document upload section
    st.header("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'doc'],
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected**")
        
        if st.button("📤 Process All Files", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")
                
                response = upload_document(uploaded_file)
                
                if response and response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {uploaded_file.name}: {result['chunks']} chunks")
                else:
                    error_msg = response.json().get("detail", "Unknown error") if response else "Connection error"
                    st.error(f"❌ {uploaded_file.name}: {error_msg}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✨ Processing complete!")
            st.rerun()
    
    st.divider()
    
    # Document management
    st.header("📚 Manage Documents")
    
    documents = get_documents_list()
    
    if documents:
        st.write(f"**{len(documents)} document(s) indexed**")
        
        with st.expander("View Documents"):
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(doc["filename"])
                with col2:
                    if st.button("🗑️", key=f"del_{doc['filename']}", help="Delete document"):
                        if delete_document(doc["filename"]):
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete")
        
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get("confirm_clear", False):
                if clear_all_documents():
                    st.success("All documents cleared!")
                    st.session_state.messages = []
                    st.session_state.confirm_clear = False
                    st.rerun()
            else:
                st.warning("Click again to confirm")
                st.session_state.confirm_clear = True
    else:
        st.info("No documents uploaded yet")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Chat with Your Documents")

with col2:
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
chat_container = st.container()

with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(f"📚 Sources ({message.get('chunks_used', 0)} chunks)"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source}**")

# Chat input
documents = get_documents_list()

if not documents:
    st.warning("⚠️ Please upload documents before asking questions.")
    prompt = None
else:
    prompt = st.chat_input(
        "Ask a question about your documents...",
        key="chat_input"
    )

# Process query
if prompt:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = query_documents(
                question=prompt,
                model=st.session_state.selected_model,
                top_k=top_k
            )
            
            if response and response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                sources = result["sources"]
                chunks_used = result["chunks_used"]
                
                # Display answer
                st.markdown(answer)
                
                # Display sources
                with st.expander(f"📚 Sources ({chunks_used} chunks)"):
                    for source in sources:
                        st.markdown(f"- **{source}**")
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "chunks_used": chunks_used
                })
            else:
                error_msg = "Failed to get response from backend"
                if response:
                    try:
                        error_detail = response.json().get("detail", error_msg)
                        error_msg = error_detail
                    except:
                        pass
                
                st.error(f"❌ {error_msg}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {error_msg}"
                })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>Powered by Ollama 🦙 | LangChain 🦜🔗 | ChromaDB 🎨 | FastAPI ⚡</p>
    <p>Backend: <a href='http://localhost:8000/docs' target='_blank'>API Documentation</a></p>
</div>
""", unsafe_allow_html=True)