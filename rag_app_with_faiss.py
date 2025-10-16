# rag_app_faiss.py - RAG Application with FAISS (Python 3.13 Compatible)
"""
RAG Application with Local LLM Support using FAISS
Requires: pip install fastapi uvicorn langchain faiss-cpu pypdf python-docx python-multipart ollama
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import pickle
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Initialize FastAPI app
app = FastAPI(title="RAG Application API with FAISS")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
UPLOAD_DIR = Path("./uploaded_documents")
FAISS_DIR = Path("./faiss_index")
UPLOAD_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)

# Current configuration
current_config = {
    "model": "llama3",
    "embedding_model": "nomic-embed-text",
    "chunk_size": 1000,
    "chunk_overlap": 200
}

# Vector store (initialized lazily)
vector_store = None
embeddings = None
document_metadata = {}  # Track documents and their chunks

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_used: int

class ModelConfig(BaseModel):
    model: str
    embedding_model: Optional[str] = None

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
    status: str

# Helper functions
def get_embeddings():
    """Get or create embeddings instance"""
    global embeddings
    if embeddings is None:
        embeddings = OllamaEmbeddings(
            model=current_config["embedding_model"]
        )
    return embeddings

def save_vector_store():
    """Save FAISS index to disk"""
    global vector_store
    if vector_store is not None:
        vector_store.save_local(str(FAISS_DIR / "index"))
        # Save metadata
        with open(FAISS_DIR / "metadata.pkl", "wb") as f:
            pickle.dump(document_metadata, f)

def load_vector_store():
    """Load FAISS index from disk"""
    global vector_store, document_metadata
    index_path = FAISS_DIR / "index"
    metadata_path = FAISS_DIR / "metadata.pkl"
    
    if index_path.exists():
        try:
            vector_store = FAISS.load_local(
                str(index_path),
                get_embeddings(),
                allow_dangerous_deserialization=True
            )
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    document_metadata = pickle.load(f)
            return True
        except Exception as e:
            print(f"Failed to load existing index: {e}")
            return False
    return False

def get_vector_store():
    """Get or create vector store instance"""
    global vector_store
    if vector_store is None:
        load_vector_store()
    return vector_store

def load_document(file_path: str):
    """Load document based on file extension"""
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load document: {str(e)}")

def process_document(file_path: str, filename: str):
    """Process document: load, split, embed, and store"""
    global vector_store
    
    # Load document
    documents = load_document(file_path)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=current_config["chunk_size"],
        chunk_overlap=current_config["chunk_overlap"],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add metadata
    for chunk in chunks:
        chunk.metadata["source"] = filename
    
    # Create or update vector store
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, get_embeddings())
    else:
        vector_store.add_documents(chunks)
    
    # Save to disk
    save_vector_store()
    
    # Update metadata
    document_metadata[filename] = len(chunks)
    
    return len(chunks)

def get_total_vectors():
    """Get total number of vectors in the store"""
    global vector_store
    if vector_store is None:
        return 0
    return sum(document_metadata.values())

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "RAG Application API with FAISS",
        "version": "1.0.0",
        "status": "running",
        "database": "FAISS"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_db": "FAISS",
        "vector_store": "initialized" if vector_store else "not initialized",
        "documents_indexed": get_total_vectors()
    }

@app.get("/models")
async def get_available_models():
    """Get list of available Ollama models"""
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return {"models": models, "current": current_config["model"]}
        else:
            # Fallback to common models
            return {
                "models": ["llama3", "mistral", "phi", "gemma", "neural-chat"],
                "current": current_config["model"],
                "note": "Could not fetch from Ollama, showing common models"
            }
    except Exception as e:
        return {
            "models": ["llama3", "mistral", "phi", "gemma", "neural-chat"],
            "current": current_config["model"],
            "error": str(e)
        }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.txt', '.docx', '.doc']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        num_chunks = process_document(str(file_path), file.filename)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": num_chunks,
            "message": f"Document processed successfully with {num_chunks} chunks"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        # Get vector store
        vs = get_vector_store()
        
        if vs is None or get_total_vectors() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        # Set model
        model_name = request.model or current_config["model"]
        
        # Create LLM
        llm = Ollama(model=model_name, temperature=0.7)
        
        # Create custom prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vs.as_retriever(
                search_kwargs={"k": request.top_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Get answer
        result = qa_chain({"query": request.question})
        
        # Extract sources
        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in result["source_documents"]
        ]))
        
        return QueryResponse(
            answer=result["result"],
            sources=sources,
            chunks_used=len(result["source_documents"])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure")
async def configure_model(config: ModelConfig):
    """Configure the RAG system"""
    global current_config, embeddings, vector_store
    
    current_config["model"] = config.model
    
    if config.embedding_model:
        current_config["embedding_model"] = config.embedding_model
        # Reset embeddings to use new model
        embeddings = None
        vector_store = None
    
    return {
        "status": "success",
        "config": current_config
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    documents = []
    for filename, chunks in document_metadata.items():
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            documents.append({
                "filename": filename,
                "size": file_path.stat().st_size,
                "chunks": chunks,
                "status": "indexed"
            })
    return {"documents": documents}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document (note: cannot remove from FAISS easily, requires rebuild)"""
    file_path = UPLOAD_DIR / filename
    
    if file_path.exists():
        file_path.unlink()
        
        # Remove from metadata
        if filename in document_metadata:
            del document_metadata[filename]
            save_vector_store()
        
        return {
            "status": "success",
            "message": f"Document {filename} deleted. Note: Vector embeddings remain until index rebuild.",
            "note": "Use /clear to completely remove all vectors"
        }
    else:
        raise HTTPException(status_code=404, detail="Document not found")

@app.delete("/clear")
async def clear_all():
    """Clear all documents and vector database"""
    global vector_store, document_metadata
    
    # Clear uploaded files
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            file_path.unlink()
    
    # Clear FAISS index
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)
        FAISS_DIR.mkdir()
    
    # Reset vector store and metadata
    vector_store = None
    document_metadata = {}
    
    return {
        "status": "success",
        "message": "All documents and embeddings cleared"
    }

# Load existing index on startup
@app.on_event("startup")
async def startup_event():
    """Load existing FAISS index on startup"""
    load_vector_store()
    print(f"Loaded {get_total_vectors()} vectors from disk")

if __name__ == "__main__":
    import uvicorn
    print("Starting RAG Application Server with FAISS...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)