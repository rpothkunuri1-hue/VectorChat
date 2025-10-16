# rag_app.py - Complete RAG Application Backend
"""
RAG Application with Local LLM Support
Requires: pip install fastapi uvicorn langchain chromadb pypdf python-docx python-multipart ollama
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import shutil
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize FastAPI app
app = FastAPI(title="RAG Application API")

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
CHROMA_DIR = Path("./chroma_db")
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

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

def get_vector_store():
    """Get or create vector store instance"""
    global vector_store
    if vector_store is None:
        vector_store = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings()
        )
    return vector_store

def load_document(file_path: str):
    """Load document based on file extension"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path)
    elif ext in ['.docx', '.doc']:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return loader.load()

def process_document(file_path: str, filename: str):
    """Process document: load, split, embed, and store"""
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
    
    # Add to vector store
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    return len(chunks)

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "RAG Application API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_db": "connected" if vector_store else "not initialized",
        "documents_indexed": get_vector_store()._collection.count() if vector_store else 0
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
        vector_store = get_vector_store()
        
        if vector_store._collection.count() == 0:
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
            retriever=vector_store.as_retriever(
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
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            documents.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "status": "indexed"
            })
    return {"documents": documents}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document"""
    file_path = UPLOAD_DIR / filename
    
    if file_path.exists():
        file_path.unlink()
        return {
            "status": "success",
            "message": f"Document {filename} deleted"
        }
    else:
        raise HTTPException(status_code=404, detail="Document not found")

@app.delete("/clear")
async def clear_all():
    """Clear all documents and vector database"""
    global vector_store
    
    # Clear uploaded files
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            file_path.unlink()
    
    # Clear vector database
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir()
    
    # Reset vector store
    vector_store = None
    
    return {
        "status": "success",
        "message": "All documents and embeddings cleared"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting RAG Application Server...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)