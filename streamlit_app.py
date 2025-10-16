# streamlit_app.py - Enhanced Interactive Streamlit UI for RAG Application
"""
Enhanced Streamlit frontend with advanced features
Run with: streamlit run streamlit_app.py
Requires: pip install streamlit requests plotly pandas
"""

import streamlit as st
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime
import time
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RAG Application - Enhanced",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.12);
    }
    
    /* Upload section */
    .upload-section {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: 2px dashed #667eea;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Status indicators */
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Document card */
    .doc-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 3px solid #667eea;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=5)
def check_backend_connection():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

@st.cache_data(ttl=10)
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

@st.cache_data(ttl=5)
def get_system_health():
    """Get system health information"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

@st.cache_data(ttl=5)
def get_statistics():
    """Get detailed statistics"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
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
        response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
        return response
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None

def query_documents(question: str, model: str, top_k: int = 4, temperature: float = 0.7):
    """Query the RAG system"""
    try:
        payload = {
            "question": question,
            "model": model,
            "top_k": top_k,
            "temperature": temperature
        }
        response = requests.post(
            f"{API_URL}/query",
            json=payload,
            timeout=90
        )
        return response
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return None

@st.cache_data(ttl=5)
def get_documents_list():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except:
        return []

def get_document_preview(filename: str, max_chunks: int = 3):
    """Get document preview"""
    try:
        response = requests.get(f"{API_URL}/documents/{filename}/preview?max_chunks={max_chunks}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

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

def update_configuration(model: str, temperature: float, chunk_size: int, chunk_overlap: int):
    """Update backend configuration"""
    try:
        payload = {
            "model": model,
            "temperature": temperature,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        response = requests.post(f"{API_URL}/configure", json=payload, timeout=5)
        return response.status_code == 200
    except:
        return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def format_timestamp(timestamp: str) -> str:
    """Format timestamp"""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "top_k" not in st.session_state:
    st.session_state.top_k = 4

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main title with gradient header
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Application - Enhanced</h1>
    <p style="font-size: 1.1rem; margin: 0;">Retrieval-Augmented Generation with Local LLMs</p>
</div>
""", unsafe_allow_html=True)

# Check backend connection
is_connected, health_data = check_backend_connection()

if not is_connected:
    st.error("⚠️ Cannot connect to backend server. Please ensure the backend is running at http://localhost:8000")
    st.info("Start the backend with: `python rag_app.py`")
    st.code("python rag_app.py", language="bash")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model selection
    available_models, current_model = get_available_models()
    selected_model = st.selectbox(
        "🦙 LLM Model",
        available_models,
        index=available_models.index(current_model) if current_model in available_models else 0,
        help="Choose the Ollama model for answer generation"
    )
    st.session_state.selected_model = selected_model
    
    # Advanced settings in expander
    with st.expander("🎛️ Advanced Settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        st.session_state.temperature = temperature
        
        top_k = st.slider(
            "Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of document chunks to retrieve for context"
        )
        st.session_state.top_k = top_k
        
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            value=200,
            step=50,
            help="Overlap between consecutive chunks"
        )
        
        if st.button("💾 Save Configuration"):
            if update_configuration(selected_model, temperature, chunk_size, chunk_overlap):
                st.success("✅ Configuration saved!")
                st.cache_data.clear()
            else:
                st.error("❌ Failed to save configuration")
    
    st.divider()
    
    # System status dashboard
    st.markdown("### 📊 System Status")
    
    health = get_system_health()
    stats = get_statistics()
    
    if health:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="status-online">🟢 Online</p>', unsafe_allow_html=True)
        with col2:
            st.metric("Vectors", health.get("documents_indexed", 0))
        
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("total_documents", 0))
                st.metric("Queries", stats.get("total_queries", 0))
            with col2:
                st.metric("Chunks", stats.get("total_chunks", 0))
                total_size = stats.get("total_size_bytes", 0)
                st.metric("Size", format_file_size(total_size))
    else:
        st.markdown('<p class="status-offline">🔴 Offline</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Document upload section
    st.markdown("### 📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'doc'],
        help="Supported: PDF, TXT, DOCX"
    )
    
    if uploaded_files:
        st.info(f"**{len(uploaded_files)} file(s) selected**")
        
        if st.button("🚀 Process All Files", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_container = st.container()
            
            success_count = 0
            error_count = 0
            
            for idx, uploaded_file in enumerate(uploaded_files):
                with status_container:
                    with st.spinner(f"Processing: {uploaded_file.name}..."):
                        response = upload_document(uploaded_file)
                        
                        if response and response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {uploaded_file.name}: {result['chunks']} chunks")
                            success_count += 1
                        else:
                            error_msg = response.json().get("detail", "Unknown error") if response else "Connection error"
                            st.error(f"❌ {uploaded_file.name}: {error_msg}")
                            error_count += 1
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        time.sleep(0.5)
            
            st.balloons()
            st.success(f"✨ Complete! Success: {success_count}, Errors: {error_count}")
            time.sleep(2)
            st.cache_data.clear()
            st.rerun()
    
    st.divider()
    
    # Document management
    st.markdown("### 📚 Document Library")
    
    documents = get_documents_list()
    
    if documents:
        st.info(f"**{len(documents)} document(s) indexed**")
        
        # Search/filter documents
        search_query = st.text_input("🔍 Search documents", placeholder="Type to filter...")
        
        filtered_docs = [doc for doc in documents if search_query.lower() in doc["filename"].lower()] if search_query else documents
        
        if filtered_docs:
            for doc in filtered_docs:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Chunks", doc.get("chunks", 0))
                        st.metric("Size", format_file_size(doc.get("size", 0)))
                    with col2:
                        st.text(f"Type: {doc.get('type', 'N/A')}")
                        st.text(f"Status: {doc.get('status', 'N/A')}")
                    
                    st.caption(f"Uploaded: {format_timestamp(doc.get('uploaded_at', ''))}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👁️ Preview", key=f"prev_{doc['filename']}", use_container_width=True):
                            preview = get_document_preview(doc['filename'])
                            if preview:
                                st.session_state[f"preview_{doc['filename']}"] = preview
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{doc['filename']}", use_container_width=True, type="secondary"):
                            if delete_document(doc['filename']):
                                st.success("Deleted!")
                                st.cache_data.clear()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete")
                    
                    # Show preview if available
                    if f"preview_{doc['filename']}" in st.session_state:
                        preview = st.session_state[f"preview_{doc['filename']}"]
                        st.markdown("**Preview:**")
                        for chunk in preview.get("preview", []):
                            st.text_area(
                                f"Chunk {chunk['chunk_id']} ({chunk['length']} chars)",
                                chunk['content'],
                                height=100,
                                disabled=True
                            )
        else:
            st.warning("No documents match your search")
        
        st.divider()
        
        # Bulk actions
        st.markdown("### 🗂️ Bulk Actions")
        if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
            if st.session_state.confirm_clear:
                with st.spinner("Clearing all documents..."):
                    if clear_all_documents():
                        st.success("All documents cleared!")
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.confirm_clear = False
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to clear documents")
            else:
                st.warning("⚠️ Click again to confirm deletion")
                st.session_state.confirm_clear = True
    else:
        st.info("📭 No documents uploaded yet")
        st.markdown("""
        **Get started:**
        1. Upload documents using the form above
        2. Wait for processing to complete
        3. Start asking questions!
        """)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "ℹ️ About"])

with tab1:
    # Chat interface
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 💬 Chat with Your Documents")
    
    with col2:
        show_sources = st.checkbox(
            "Show Sources",
            value=st.session_state.show_sources,
            help="Display source documents in responses"
        )
        st.session_state.show_sources = show_sources
    
    with col3:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    
                    # Show processing time and similarity scores
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"⏱️ {metadata.get('processing_time', 0):.2f}s")
                    with col2:
                        st.caption(f"📊 {metadata.get('chunks_used', 0)} chunks")
                    with col3:
                        avg_score = sum(metadata.get('similarity_scores', [0])) / len(metadata.get('similarity_scores', [1]))
                        st.caption(f"🎯 {avg_score:.2%} relevance")
                    
                    # Display sources
                    if show_sources and "sources" in metadata:
                        with st.expander(f"📚 Sources ({len(metadata['sources'])})"):
                            for i, source in enumerate(metadata["sources"], 1):
                                score = metadata.get('similarity_scores', [])[i-1] if i-1 < len(metadata.get('similarity_scores', [])) else 0
                                st.markdown(f"{i}. **{source}** (Relevance: {score:.2%})")
    
    # Chat input
    documents = get_documents_list()
    
    if not documents:
        st.warning("⚠️ Please upload documents before asking questions.")
        st.info("Use the sidebar to upload PDF, TXT, or DOCX files.")
    else:
        prompt = st.chat_input(
            "Ask a question about your documents...",
            key="chat_input"
        )
        
        if prompt:
            # Add user message
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
                        top_k=st.session_state.top_k,
                        temperature=st.session_state.temperature
                    )
                    
                    if response and response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        sources = result["sources"]
                        chunks_used = result["chunks_used"]
                        similarity_scores = result.get("similarity_scores", [])
                        processing_time = result.get("processing_time", 0)
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Display metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"⏱️ {processing_time:.2f}s")
                        with col2:
                            st.caption(f"📊 {chunks_used} chunks")
                        with col3:
                            avg_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
                            st.caption(f"🎯 {avg_score:.2%} relevance")
                        
                        # Display sources
                        if show_sources:
                            with st.expander(f"📚 Sources ({len(sources)})"):
                                for i, source in enumerate(sources, 1):
                                    score = similarity_scores[i-1] if i-1 < len(similarity_scores) else 0
                                    st.markdown(f"{i}. **{source}** (Relevance: {score:.2%})")
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "metadata": {
                                "sources": sources,
                                "chunks_used": chunks_used,
                                "similarity_scores": similarity_scores,
                                "processing_time": processing_time
                            }
                        })
                        
                        # Save to chat history
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "question": prompt,
                            "answer": answer,
                            "sources": sources,
                            "processing_time": processing_time
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

with tab2:
    # Analytics dashboard
    st.markdown("### 📊 Analytics Dashboard")
    
    stats = get_statistics()
    documents = get_documents_list()
    
    if stats and documents:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Documents",
                stats.get("total_documents", 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Chunks",
                stats.get("total_chunks", 0),
                delta=None
            )
        
        with col3:
            st.metric(
                "Total Queries",
                stats.get("total_queries", 0),
                delta=None
            )
        
        with col4:
            total_size = stats.get("total_size_bytes", 0)
            st.metric(
                "Total Size",
                format_file_size(total_size),
                delta=None
            )
        
        st.divider()
        
        # Document breakdown
        st.markdown("#### 📄 Document Breakdown")
        
        if documents:
            # Create dataframe
            df_docs = pd.DataFrame([
                {
                    "Filename": doc["filename"],
                    "Chunks": doc.get("chunks", 0),
                    "Size": format_file_size(doc.get("size", 0)),
                    "Type": doc.get("type", "N/A"),
                    "Status": doc.get("status", "N/A"),
                    "Uploaded": format_timestamp(doc.get("uploaded_at", ""))
                }
                for doc in documents
            ])
            
            st.dataframe(
                df_docs,
                use_container_width=True,
                hide_index=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Chunks per Document")
                chart_data = pd.DataFrame({
                    "Document": [doc["filename"][:20] + "..." if len(doc["filename"]) > 20 else doc["filename"] for doc in documents],
                    "Chunks": [doc.get("chunks", 0) for doc in documents]
                })
                st.bar_chart(chart_data.set_index("Document"))
            
            with col2:
                st.markdown("##### File Type Distribution")
                type_counts = {}
                for doc in documents:
                    file_type = doc.get("type", "unknown")
                    type_counts[file_type] = type_counts.get(file_type, 0) + 1
                
                type_df = pd.DataFrame(list(type_counts.items()), columns=["Type", "Count"])
                st.bar_chart(type_df.set_index("Type"))
        
        st.divider()
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("#### 💬 Recent Chat History")
            
            # Convert to dataframe
            df_history = pd.DataFrame([
                {
                    "Time": format_timestamp(chat["timestamp"]),
                    "Question": chat["question"][:50] + "..." if len(chat["question"]) > 50 else chat["question"],
                    "Sources": ", ".join(chat["sources"]),
                    "Processing Time": f"{chat['processing_time']:.2f}s"
                }
                for chat in st.session_state.chat_history[-10:]  # Last 10 chats
            ])
            
            st.dataframe(
                df_history,
                use_container_width=True,
                hide_index=True
            )
            
            # Export chat history
            if st.button("📥 Export Chat History"):
                history_json = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=history_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.info("📊 No data available yet. Upload documents and start chatting to see analytics!")

with tab3:
    # About section
    st.markdown("### ℹ️ About This Application")
    
    st.markdown("""
    This is an enhanced **Retrieval-Augmented Generation (RAG)** application that allows you to:
    
    #### ✨ Features
    - 📤 Upload multiple documents (PDF, TXT, DOCX)
    - 🤖 Ask questions using local LLMs via Ollama
    - 🔍 Retrieve relevant context from your documents
    - 💬 Interactive chat interface with history
    - 📊 Real-time analytics and statistics
    - ⚙️ Configurable parameters and settings
    - 👁️ Document preview and management
    - 🎯 Similarity scores and relevance metrics
    
    #### 🛠️ Technology Stack
    - **Backend**: FastAPI, LangChain, Ollama
    - **Frontend**: Streamlit
    - **Vector Store**: In-memory with scikit-learn
    - **Embeddings**: Nomic Embed Text
    - **LLMs**: Llama 3, Mistral, Phi, Gemma
    
    #### 📖 How to Use
    1. **Upload Documents**: Use the sidebar to upload your files
    2. **Configure Settings**: Adjust model, temperature, and retrieval parameters
    3. **Ask Questions**: Type your questions in the chat interface
    4. **View Analytics**: Check the Analytics tab for insights
    
    #### 🔗 Links
    - Backend API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
    - Health Check: [http://localhost:8000/health](http://localhost:8000/health)
    """)
    
    st.divider()
    
    # System information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🖥️ System Info")
        if health_data:
            st.json(health_data)
    
    with col2:
        st.markdown("#### ⚙️ Current Configuration")
        if stats:
            config_data = stats.get("config", {})
            st.json(config_data)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;'>
    <p><strong>Powered by:</strong> Ollama 🦙 | LangChain 🦜🔗 | FastAPI ⚡ | Streamlit 🎈</p>
    <p>Made with ❤️ for document Q&A</p>
</div>
""", unsafe_allow_html=True)
