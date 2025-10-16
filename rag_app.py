# rag_app_simple.py - RAG Application without FAISS/ChromaDB
"""
Simplified RAG Application using in-memory vector storage
NO BUILD DEPENDENCIES REQUIRED - Pure Python
Requires: pip install fastapi uvicorn langchain langchain-community pypdf python-docx python-multipart ollama docx2txt numpy scikit-learn
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Initialize FastAPI app
app = FastAPI(title="RAG Application API - Simple")

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
DATA_DIR = Path("./vector_data")
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Current configuration
current_config = {
    "model": "llama3",
    "embedding_model": "nomic-embed-text",
    "chunk_size": 1000,
    "chunk_overlap": 200
}

# In-memory vector store
class SimpleVectorStore:
    def __init__(self):
        self.embeddings_list = []
        self.documents = []
        self.metadata = []
        
    def add_documents(self, docs, embeddings):
        """Add documents with their embeddings"""
        self.documents.extend(docs)
        self.embeddings_list.extend(embeddings)
        self.metadata.extend([doc.metadata for doc in docs])
        
    def similarity_search(self, query_embedding, k=4):
        """Find most similar documents"""
        if not self.embeddings_list:
            return []
        
        # Calculate cosine similarity
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings_list
        )[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return documents
        return [self.documents[i] for i in top_indices]
    
    def get_count(self):
        """Get number of documents"""
        return len(self.documents)
    
    def save(self):
        """Save to disk"""
        data = {
            'embeddings': [emb.tolist() for emb in self.embeddings_list],
            'documents': [{'page_content': doc.page_content, 'metadata': doc.metadata} 
                         for doc in self.documents],
            'metadata': self.metadata
        }
        with open(DATA_DIR / 'vectors.json', 'w') as f:
            json.dump(data, f)
    
    def load(self):
        """Load from disk"""
        try:
            with open(DATA_DIR / 'vectors.json', 'r') as f:
                data = json.load(f)
            
            from langchain.docstore.document import Document
            self.embeddings_list = [np.array(emb) for emb in data['embeddings']]
            self.documents = [
                Document(page_content=doc['page_content'], metadata=doc['metadata'])
                for doc in data['documents']
            ]
            self.metadata = data['metadata']
            return True
        except:
            return False
    
    def clear(self):
        """Clear all data"""
        self.embeddings_list = []
        self.documents = []
        self.metadata = []

# Global instances
vector_store = SimpleVectorStore()
embeddings = None
document_metadata = {}

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

# Helper functions
def get_embeddings():
    """Get or create embeddings instance"""
    global embeddings
    if embeddings is None:
        embeddings = OllamaEmbeddings(
            model=current_config["embedding_model"]
        )
    return embeddings

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
    
    # Generate embeddings
    embeddings_model = get_embeddings()
    texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embeddings_model.embed_documents(texts)
    
    # Add to vector store
    vector_store.add_documents(chunks, chunk_embeddings)
    vector_store.save()
    
    # Update metadata
    document_metadata[filename] = len(chunks)
    
    return len(chunks)

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "RAG Application API - Simple Vector Store",
        "version": "1.0.0",
        "status": "running",
        "database": "In-Memory + File Storage"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_db": "SimpleVectorStore",
        "documents_indexed": vector_store.get_count()
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
            lines = result.stdout.strip().split('\n')[1:]
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return {"models": models, "current": current_config["model"]}
        else:
            return {
                "models": ["llama3", "mistral", "phi", "gemma"],
                "current": current_config["model"]
            }
    except Exception as e:
        return {
            "models": ["llama3", "mistral", "phi", "gemma"],
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
        if vector_store.get_count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        # Set model
        model_name = request.model or current_config["model"]
        
        # Get query embedding
        embeddings_model = get_embeddings()
        query_embedding = embeddings_model.embed_query(request.question)
        
        # Search for similar documents
        similar_docs = vector_store.similarity_search(query_embedding, k=request.top_k)
        
        if not similar_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found"
            )
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in similar_docs])
        
        # Create LLM
        llm = Ollama(model=model_name, temperature=0.7)
        
        # Create prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
        
        prompt = prompt_template.format(context=context, question=request.question)
        
        # Get answer
        answer = llm.invoke(prompt)
        
        # Extract sources
        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in similar_docs
        ]))
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(similar_docs)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure")
async def configure_model(config: ModelConfig):
    """Configure the RAG system"""
    global current_config, embeddings
    
    current_config["model"] = config.model
    
    if config.embedding_model:
        current_config["embedding_model"] = config.embedding_model
        embeddings = None
    
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
    """Delete a document"""
    file_path = UPLOAD_DIR / filename
    
    if file_path.exists():
        file_path.unlink()
        if filename in document_metadata:
            del document_metadata[filename]
        
        return {
            "status": "success",
            "message": f"Document {filename} deleted",
            "note": "Vector store rebuild recommended"
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
    
    # Clear vector data
    if (DATA_DIR / 'vectors.json').exists():
        (DATA_DIR / 'vectors.json').unlink()
    
    # Reset
    vector_store.clear()
    document_metadata = {}
    
    return {
        "status": "success",
        "message": "All documents and embeddings cleared"
    }

@app.on_event("startup")
async def startup_event():
    """Load existing data on startup"""
    vector_store.load()
    print(f"Loaded {vector_store.get_count()} document chunks")

if __name__ == "__main__":
    import uvicorn
    print("Starting RAG Application Server (Simple Vector Store)...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)