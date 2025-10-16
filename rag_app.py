# rag_app.py - Enhanced RAG Application Backend
"""
Enhanced RAG Application with improved error handling, logging, and features
Requires: pip install fastapi uvicorn langchain langchain-community pypdf python-docx 
          python-multipart ollama docx2txt numpy scikit-learn
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import shutil
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.schema import Document

# Global variables
UPLOAD_DIR = Path("./uploaded_documents")
DATA_DIR = Path("./vector_data")
METADATA_FILE = DATA_DIR / "metadata.json"

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Configuration
class AppConfig:
    def __init__(self):
        self.model = "llama3"
        self.embedding_model = "nomic-embed-text"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.temperature = 0.7
        
    def to_dict(self):
        return {
            "model": self.model,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "temperature": self.temperature
        }

config = AppConfig()

# Enhanced Vector Store
class SimpleVectorStore:
    def __init__(self):
        self.embeddings_list = []
        self.documents = []
        self.metadata = []
        self.stats = {
            "total_chunks": 0,
            "total_queries": 0,
            "last_updated": None
        }
        
    def add_documents(self, docs: List[Document], embeddings: List[np.ndarray]):
        """Add documents with their embeddings"""
        self.documents.extend(docs)
        self.embeddings_list.extend(embeddings)
        self.metadata.extend([doc.metadata for doc in docs])
        self.stats["total_chunks"] = len(self.documents)
        self.stats["last_updated"] = datetime.now().isoformat()
        
    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> List[Document]:
        """Find most similar documents"""
        if not self.embeddings_list:
            return []
        
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings_list
        )[0]
        
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for i in top_indices:
            doc = self.documents[i]
            doc.metadata["similarity_score"] = float(similarities[i])
            results.append(doc)
        
        self.stats["total_queries"] += 1
        return results
    
    def get_count(self) -> int:
        return len(self.documents)
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()
    
    def get_documents_by_source(self, source: str) -> List[Document]:
        """Get all documents from a specific source"""
        return [doc for doc in self.documents if doc.metadata.get("source") == source]
    
    def save(self):
        """Save to disk"""
        try:
            data = {
                'embeddings': [emb.tolist() for emb in self.embeddings_list],
                'documents': [
                    {
                        'page_content': doc.page_content, 
                        'metadata': doc.metadata
                    } 
                    for doc in self.documents
                ],
                'metadata': self.metadata,
                'stats': self.stats
            }
            with open(DATA_DIR / 'vectors.json', 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.documents)} documents to disk")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load(self) -> bool:
        """Load from disk"""
        try:
            vector_file = DATA_DIR / 'vectors.json'
            if not vector_file.exists():
                return False
                
            with open(vector_file, 'r') as f:
                data = json.load(f)
            
            self.embeddings_list = [np.array(emb) for emb in data['embeddings']]
            self.documents = [
                Document(page_content=doc['page_content'], metadata=doc['metadata'])
                for doc in data['documents']
            ]
            self.metadata = data['metadata']
            self.stats = data.get('stats', self.stats)
            
            logger.info(f"Loaded {len(self.documents)} documents from disk")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def clear(self):
        """Clear all data"""
        self.embeddings_list = []
        self.documents = []
        self.metadata = []
        self.stats = {
            "total_chunks": 0,
            "total_queries": 0,
            "last_updated": None
        }

# Document Metadata Manager
class DocumentMetadataManager:
    def __init__(self):
        self.metadata = {}
        self.load()
    
    def add_document(self, filename: str, chunks: int, file_size: int, file_type: str):
        self.metadata[filename] = {
            "chunks": chunks,
            "size": file_size,
            "type": file_type,
            "uploaded_at": datetime.now().isoformat(),
            "status": "indexed"
        }
        self.save()
    
    def remove_document(self, filename: str):
        if filename in self.metadata:
            del self.metadata[filename]
            self.save()
    
    def get_document(self, filename: str) -> Optional[Dict]:
        return self.metadata.get(filename)
    
    def get_all(self) -> Dict:
        return self.metadata.copy()
    
    def clear(self):
        self.metadata = {}
        self.save()
    
    def save(self):
        try:
            with open(METADATA_FILE, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def load(self):
        try:
            if METADATA_FILE.exists():
                with open(METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {}

# Global instances
vector_store = SimpleVectorStore()
doc_metadata = DocumentMetadataManager()
embeddings_cache = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    model: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=10)
    temperature: Optional[float] = Field(default=None, ge=0, le=1)

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_used: int
    similarity_scores: List[float]
    processing_time: float

class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int
    file_size: int
    message: str

class ModelConfig(BaseModel):
    model: str
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=100, le=5000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000)
    temperature: Optional[float] = Field(default=None, ge=0, le=1)

class DocumentInfo(BaseModel):
    filename: str
    size: int
    chunks: int
    status: str
    uploaded_at: str
    type: str

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RAG Application Server...")
    vector_store.load()
    logger.info(f"Loaded {vector_store.get_count()} document chunks")
    yield
    # Shutdown
    logger.info("Shutting down RAG Application Server...")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Application API",
    version="2.0.0",
    description="Enhanced RAG Application with Simple Vector Store",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def get_embeddings():
    """Get or create embeddings instance"""
    global embeddings_cache
    if embeddings_cache is None:
        try:
            embeddings_cache = OllamaEmbeddings(
                model=config.embedding_model
            )
            logger.info(f"Initialized embeddings with model: {config.embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize embeddings: {str(e)}")
    return embeddings_cache

def load_document(file_path: str) -> List[Document]:
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
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load document {file_path}: {e}")
        raise ValueError(f"Failed to load document: {str(e)}")

def process_document(file_path: str, filename: str) -> int:
    """Process document: load, split, embed, and store"""
    try:
        documents = load_document(file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = filename
            chunk.metadata["chunk_id"] = i
        
        embeddings_model = get_embeddings()
        texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = embeddings_model.embed_documents(texts)
        
        vector_store.add_documents(chunks, chunk_embeddings)
        vector_store.save()
        
        logger.info(f"Processed {filename}: {len(chunks)} chunks created")
        return len(chunks)
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        raise

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "RAG Application API - Enhanced Version",
        "version": "2.0.0",
        "status": "running",
        "database": "In-Memory + File Storage",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "vector_db": "SimpleVectorStore",
        "documents_indexed": vector_store.get_count(),
        "total_queries": stats.get("total_queries", 0),
        "last_updated": stats.get("last_updated"),
        "config": config.to_dict()
    }

@app.get("/models")
async def get_available_models():
    """Get list of available Ollama models"""
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return {
                "models": models if models else ["llama3", "mistral", "phi"],
                "current": config.model
            }
        else:
            return {
                "models": ["llama3", "mistral", "phi", "gemma"],
                "current": config.model
            }
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {
            "models": ["llama3", "mistral", "phi", "gemma"],
            "current": config.model,
            "error": str(e)
        }

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        allowed_extensions = ['.pdf', '.txt', '.docx', '.doc']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        file_path = UPLOAD_DIR / file.filename
        
        # Check if file already exists
        if file_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' already exists. Please delete it first or rename the file."
            )
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            file_size = len(content)
        
        # Process document
        num_chunks = process_document(str(file_path), file.filename)
        
        # Update metadata
        doc_metadata.add_document(
            filename=file.filename,
            chunks=num_chunks,
            file_size=file_size,
            file_type=file_ext
        )
        
        return DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=num_chunks,
            file_size=file_size,
            message=f"Document processed successfully with {num_chunks} chunks"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        # Clean up file if it was created
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    start_time = datetime.now()
    
    try:
        if vector_store.get_count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        model_name = request.model or config.model
        temperature = request.temperature if request.temperature is not None else config.temperature
        
        embeddings_model = get_embeddings()
        query_embedding = embeddings_model.embed_query(request.question)
        
        similar_docs = vector_store.similarity_search(query_embedding, k=request.top_k)
        
        if not similar_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found"
            )
        
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in similar_docs
        ])
        
        llm = Ollama(model=model_name, temperature=temperature)
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Provide a clear, concise answer based solely on the given context.

Context:
{context}

Question: {question}

Answer: """
        
        prompt = prompt_template.format(context=context, question=request.question)
        answer = llm.invoke(prompt)
        
        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in similar_docs
        ]))
        
        similarity_scores = [
            doc.metadata.get("similarity_score", 0.0)
            for doc in similar_docs
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Query processed in {processing_time:.2f}s - Question: {request.question[:50]}...")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(similar_docs),
            similarity_scores=similarity_scores,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure")
async def configure_model(new_config: ModelConfig):
    """Configure the RAG system"""
    global embeddings_cache
    
    try:
        config.model = new_config.model
        
        if new_config.embedding_model:
            config.embedding_model = new_config.embedding_model
            embeddings_cache = None
        
        if new_config.chunk_size:
            config.chunk_size = new_config.chunk_size
        
        if new_config.chunk_overlap:
            config.chunk_overlap = new_config.chunk_overlap
        
        if new_config.temperature is not None:
            config.temperature = new_config.temperature
        
        logger.info(f"Configuration updated: {config.to_dict()}")
        
        return {
            "status": "success",
            "config": config.to_dict()
        }
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=Dict[str, List[DocumentInfo]])
async def list_documents():
    """List all uploaded documents"""
    documents = []
    all_metadata = doc_metadata.get_all()
    
    for filename, meta in all_metadata.items():
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            documents.append(DocumentInfo(
                filename=filename,
                size=meta.get("size", 0),
                chunks=meta.get("chunks", 0),
                status=meta.get("status", "unknown"),
                uploaded_at=meta.get("uploaded_at", ""),
                type=meta.get("type", "")
            ))
    
    return {"documents": documents}

@app.get("/documents/{filename}/preview")
async def preview_document(filename: str, max_chunks: int = 3):
    """Preview document chunks"""
    try:
        chunks = vector_store.get_documents_by_source(filename)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        preview_chunks = chunks[:max_chunks]
        
        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "preview": [
                {
                    "chunk_id": chunk.metadata.get("chunk_id", i),
                    "content": chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
                    "length": len(chunk.page_content)
                }
                for i, chunk in enumerate(preview_chunks)
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document"""
    try:
        file_path = UPLOAD_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_path.unlink()
        doc_metadata.remove_document(filename)
        
        logger.info(f"Deleted document: {filename}")
        
        return {
            "status": "success",
            "message": f"Document '{filename}' deleted successfully",
            "note": "Vector store rebuild recommended for complete removal"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_all():
    """Clear all documents and vector database"""
    try:
        # Clear uploaded files
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
        
        # Clear vector data
        vector_file = DATA_DIR / 'vectors.json'
        if vector_file.exists():
            vector_file.unlink()
        
        # Reset
        vector_store.clear()
        doc_metadata.clear()
        
        logger.info("All documents and embeddings cleared")
        
        return {
            "status": "success",
            "message": "All documents and embeddings cleared successfully"
        }
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get detailed statistics"""
    stats = vector_store.get_stats()
    all_docs = doc_metadata.get_all()
    
    total_size = sum(doc.get("size", 0) for doc in all_docs.values())
    
    return {
        "total_documents": len(all_docs),
        "total_chunks": stats.get("total_chunks", 0),
        "total_queries": stats.get("total_queries", 0),
        "total_size_bytes": total_size,
        "last_updated": stats.get("last_updated"),
        "config": config.to_dict()
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Enhanced RAG Application Server...")
    print("=" * 60)
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
