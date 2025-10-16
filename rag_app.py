# rag_app.py - Enhanced RAG Application Backend with Comprehensive Debugging
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
import traceback
import sys

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_app.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("RAG Application Starting...")
logger.info("=" * 80)

# LangChain imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
        PyPDFLoader, 
        TextLoader,
        Docx2txtLoader
    )
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
    from langchain.schema import Document
    logger.info("✓ All LangChain imports successful")
except Exception as e:
    logger.error(f"✗ Failed to import LangChain modules: {e}")
    logger.error(traceback.format_exc())
    raise

# Global variables
UPLOAD_DIR = Path("./uploaded_documents")
DATA_DIR = Path("./vector_data")
METADATA_FILE = DATA_DIR / "metadata.json"

logger.info(f"Upload directory: {UPLOAD_DIR.absolute()}")
logger.info(f"Data directory: {DATA_DIR.absolute()}")

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
logger.info("✓ Directories created/verified")

# Configuration
class AppConfig:
    def __init__(self):
        self.model = "llama3"
        self.embedding_model = "nomic-embed-text"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.temperature = 0.7
        logger.debug(f"AppConfig initialized: {self.to_dict()}")
        
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
        logger.debug("Initializing SimpleVectorStore")
        self.embeddings_list = []
        self.documents = []
        self.metadata = []
        self.stats = {
            "total_chunks": 0,
            "total_queries": 0,
            "last_updated": None
        }
        logger.info("✓ SimpleVectorStore initialized")
        
    def add_documents(self, docs: List[Document], embeddings: List[np.ndarray]):
        """Add documents with their embeddings"""
        logger.debug(f"Adding {len(docs)} documents to vector store")
        try:
            self.documents.extend(docs)
            self.embeddings_list.extend(embeddings)
            self.metadata.extend([doc.metadata for doc in docs])
            self.stats["total_chunks"] = len(self.documents)
            self.stats["last_updated"] = datetime.now().isoformat()
            logger.info(f"✓ Added {len(docs)} documents. Total chunks: {self.stats['total_chunks']}")
        except Exception as e:
            logger.error(f"✗ Failed to add documents: {e}")
            logger.error(traceback.format_exc())
            raise
        
    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> List[Document]:
        """Find most similar documents"""
        logger.debug(f"Performing similarity search with k={k}")
        
        if not self.embeddings_list:
            logger.warning("No embeddings in vector store")
            return []
        
        try:
            similarities = cosine_similarity(
                [query_embedding],
                self.embeddings_list
            )[0]
            
            top_indices = np.argsort(similarities)[-k:][::-1]
            logger.debug(f"Top {k} similarity scores: {similarities[top_indices]}")
            
            results = []
            for i in top_indices:
                doc = self.documents[i]
                doc.metadata["similarity_score"] = float(similarities[i])
                results.append(doc)
            
            self.stats["total_queries"] += 1
            logger.info(f"✓ Similarity search complete. Returned {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"✗ Similarity search failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_count(self) -> int:
        count = len(self.documents)
        logger.debug(f"Vector store contains {count} documents")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()
    
    def get_documents_by_source(self, source: str) -> List[Document]:
        """Get all documents from a specific source"""
        logger.debug(f"Getting documents by source: {source}")
        docs = [doc for doc in self.documents if doc.metadata.get("source") == source]
        logger.debug(f"Found {len(docs)} documents for source: {source}")
        return docs
    
    def save(self):
        """Save to disk"""
        logger.info("Saving vector store to disk...")
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
            
            save_path = DATA_DIR / 'vectors.json'
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"✓ Saved {len(self.documents)} documents to {save_path}")
        except Exception as e:
            logger.error(f"✗ Error saving vector store: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def load(self) -> bool:
        """Load from disk"""
        logger.info("Loading vector store from disk...")
        try:
            vector_file = DATA_DIR / 'vectors.json'
            if not vector_file.exists():
                logger.warning(f"Vector file not found: {vector_file}")
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
            
            logger.info(f"✓ Loaded {len(self.documents)} documents from {vector_file}")
            return True
        except Exception as e:
            logger.error(f"✗ Error loading vector store: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def clear(self):
        """Clear all data"""
        logger.info("Clearing vector store...")
        self.embeddings_list = []
        self.documents = []
        self.metadata = []
        self.stats = {
            "total_chunks": 0,
            "total_queries": 0,
            "last_updated": None
        }
        logger.info("✓ Vector store cleared")

# Document Metadata Manager
class DocumentMetadataManager:
    def __init__(self):
        logger.debug("Initializing DocumentMetadataManager")
        self.metadata = {}
        self.load()
        logger.info("✓ DocumentMetadataManager initialized")
    
    def add_document(self, filename: str, chunks: int, file_size: int, file_type: str):
        logger.debug(f"Adding document metadata: {filename}")
        self.metadata[filename] = {
            "chunks": chunks,
            "size": file_size,
            "type": file_type,
            "uploaded_at": datetime.now().isoformat(),
            "status": "indexed"
        }
        self.save()
        logger.info(f"✓ Added metadata for {filename}")
    
    def remove_document(self, filename: str):
        logger.debug(f"Removing document metadata: {filename}")
        if filename in self.metadata:
            del self.metadata[filename]
            self.save()
            logger.info(f"✓ Removed metadata for {filename}")
    
    def get_document(self, filename: str) -> Optional[Dict]:
        return self.metadata.get(filename)
    
    def get_all(self) -> Dict:
        return self.metadata.copy()
    
    def clear(self):
        logger.info("Clearing all document metadata")
        self.metadata = {}
        self.save()
        logger.info("✓ Metadata cleared")
    
    def save(self):
        try:
            with open(METADATA_FILE, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"✓ Metadata saved to {METADATA_FILE}")
        except Exception as e:
            logger.error(f"✗ Error saving metadata: {e}")
            logger.error(traceback.format_exc())
    
    def load(self):
        try:
            if METADATA_FILE.exists():
                with open(METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)
                logger.debug(f"✓ Loaded metadata from {METADATA_FILE}")
            else:
                logger.debug(f"Metadata file not found: {METADATA_FILE}")
        except Exception as e:
            logger.error(f"✗ Error loading metadata: {e}")
            logger.error(traceback.format_exc())
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
    logger.info("=" * 80)
    logger.info("FastAPI Application Startup")
    logger.info("=" * 80)
    try:
        vector_store.load()
        logger.info(f"✓ Loaded {vector_store.get_count()} document chunks")
    except Exception as e:
        logger.error(f"✗ Failed to load vector store: {e}")
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("FastAPI Application Shutdown")
    logger.info("=" * 80)

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
logger.info("✓ CORS middleware configured")

# Helper functions
def get_embeddings():
    """Get or create embeddings instance"""
    global embeddings_cache
    logger.debug("Getting embeddings instance")
    
    if embeddings_cache is None:
        try:
            logger.info(f"Initializing OllamaEmbeddings with model: {config.embedding_model}")
            embeddings_cache = OllamaEmbeddings(
                model=config.embedding_model
            )
            logger.info(f"✓ Embeddings initialized with model: {config.embedding_model}")
        except Exception as e:
            logger.error(f"✗ Failed to initialize embeddings: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize embeddings: {str(e)}"
            )
    return embeddings_cache

def load_document(file_path: str) -> List[Document]:
    """Load document based on file extension"""
    logger.info(f"Loading document: {file_path}")
    ext = Path(file_path).suffix.lower()
    logger.debug(f"File extension: {ext}")
    
    try:
        if ext == '.pdf':
            logger.debug("Using PyPDFLoader")
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            logger.debug("Using TextLoader")
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext in ['.docx', '.doc']:
            logger.debug("Using Docx2txtLoader")
            loader = Docx2txtLoader(file_path)
        else:
            logger.error(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")
        
        documents = loader.load()
        logger.info(f"✓ Loaded {len(documents)} pages from {file_path}")
        
        # Log first 200 chars of first document for debugging
        if documents:
            preview = documents[0].page_content[:200]
            logger.debug(f"First document preview: {preview}...")
        
        return documents
    except Exception as e:
        logger.error(f"✗ Failed to load document {file_path}: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to load document: {str(e)}")

def process_document(file_path: str, filename: str) -> int:
    """Process document: load, split, embed, and store"""
    logger.info("=" * 60)
    logger.info(f"Processing document: {filename}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load document
        logger.info("STEP 1: Loading document...")
        documents = load_document(file_path)
        logger.info(f"✓ Loaded {len(documents)} page(s)")
        
        # Step 2: Split into chunks
        logger.info("STEP 2: Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )
        logger.debug(f"Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        # Step 3: Add metadata
        logger.info("STEP 3: Adding metadata to chunks...")
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = filename
            chunk.metadata["chunk_id"] = i
            if i == 0:
                logger.debug(f"Sample chunk metadata: {chunk.metadata}")
        logger.info(f"✓ Metadata added to all chunks")
        
        # Step 4: Generate embeddings
        logger.info("STEP 4: Generating embeddings...")
        embeddings_model = get_embeddings()
        texts = [chunk.page_content for chunk in chunks]
        logger.debug(f"Generating embeddings for {len(texts)} text chunks")
        
        chunk_embeddings = embeddings_model.embed_documents(texts)
        logger.info(f"✓ Generated {len(chunk_embeddings)} embeddings")
        logger.debug(f"Embedding dimension: {len(chunk_embeddings[0]) if chunk_embeddings else 0}")
        
        # Step 5: Store in vector database
        logger.info("STEP 5: Storing in vector database...")
        vector_store.add_documents(chunks, chunk_embeddings)
        
        # Step 6: Save to disk
        logger.info("STEP 6: Saving to disk...")
        vector_store.save()
        
        logger.info("=" * 60)
        logger.info(f"✓ SUCCESSFULLY processed {filename}: {len(chunks)} chunks")
        logger.info("=" * 60)
        
        return len(chunks)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ FAILED to process document {filename}")
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        raise

# API Endpoints

@app.get("/")
async def root():
    logger.debug("Root endpoint called")
    return {
        "message": "RAG Application API - Enhanced Version",
        "version": "2.0.0",
        "status": "running",
        "database": "In-Memory + File Storage",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint called")
    try:
        stats = vector_store.get_stats()
        response = {
            "status": "healthy",
            "vector_db": "SimpleVectorStore",
            "documents_indexed": vector_store.get_count(),
            "total_queries": stats.get("total_queries", 0),
            "last_updated": stats.get("last_updated"),
            "config": config.to_dict()
        }
        logger.debug(f"Health check response: {response}")
        return response
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.error(traceback.format_exc())
        raise

@app.get("/models")
async def get_available_models():
    """Get list of available Ollama models"""
    logger.info("Getting available models...")
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            logger.debug(f"Ollama list output: {result.stdout}")
            lines = result.stdout.strip().split('\n')[1:]
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            logger.info(f"✓ Found {len(models)} models")
            return {
                "models": models if models else ["llama3", "mistral", "phi"],
                "current": config.model
            }
        else:
            logger.warning(f"Ollama list failed: {result.stderr}")
            return {
                "models": ["llama3", "mistral", "phi", "gemma"],
                "current": config.model
            }
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        logger.error(traceback.format_exc())
        return {
            "models": ["llama3", "mistral", "phi", "gemma"],
            "current": config.model,
            "error": str(e)
        }

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    logger.info("=" * 80)
    logger.info("UPLOAD ENDPOINT CALLED")
    logger.info("=" * 80)
    
    file_path = None
    
    try:
        # Log incoming file details
        logger.info(f"File received: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        logger.debug(f"File object: {file}")
        
        # Validate file
        if not file.filename:
            logger.error("No filename provided")
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        allowed_extensions = ['.pdf', '.txt', '.docx', '.doc']
        file_ext = Path(file.filename).suffix.lower()
        logger.info(f"File extension: {file_ext}")
        
        if file_ext not in allowed_extensions:
            logger.error(f"Invalid file extension: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_ext}' not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        logger.info("✓ File extension valid")
        
        # Check if file already exists
        file_path = UPLOAD_DIR / file.filename
        logger.info(f"Target file path: {file_path.absolute()}")
        
        if file_path.exists():
            logger.error(f"File already exists: {file_path}")
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' already exists. Please delete it first or rename the file."
            )
        
        logger.info("✓ File does not exist, proceeding with upload")
        
        # Read and save file
        logger.info("Reading file content...")
        try:
            content = await file.read()
            file_size = len(content)
            logger.info(f"✓ File read successfully. Size: {file_size} bytes ({file_size / 1024:.2f} KB)")
            
            if file_size == 0:
                logger.error("File is empty")
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
        except Exception as e:
            logger.error(f"✗ Failed to read file: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        
        # Write file to disk
        logger.info(f"Writing file to: {file_path}")
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            logger.info(f"✓ File written successfully to {file_path}")
            
            # Verify file was written
            if not file_path.exists():
                logger.error("File was not written to disk")
                raise Exception("File was not written to disk")
            
            actual_size = file_path.stat().st_size
            logger.info(f"✓ File verified on disk. Size: {actual_size} bytes")
            
        except Exception as e:
            logger.error(f"✗ Failed to write file: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Process document
        logger.info("Starting document processing...")
        try:
            num_chunks = process_document(str(file_path), file.filename)
            logger.info(f"✓ Document processed: {num_chunks} chunks created")
            
        except Exception as e:
            logger.error(f"✗ Document processing failed: {e}")
            logger.error(traceback.format_exc())
            # Clean up file on processing failure
            if file_path.exists():
                logger.info(f"Cleaning up file: {file_path}")
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        
        # Update metadata
        logger.info("Updating document metadata...")
        try:
            doc_metadata.add_document(
                filename=file.filename,
                chunks=num_chunks,
                file_size=file_size,
                file_type=file_ext
            )
            logger.info("✓ Metadata updated")
            
        except Exception as e:
            logger.error(f"✗ Failed to update metadata: {e}")
            logger.error(traceback.format_exc())
        
        # Prepare response
        response = DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=num_chunks,
            file_size=file_size,
            message=f"Document processed successfully with {num_chunks} chunks"
        )
        
        logger.info("=" * 80)
        logger.info(f"✓ QUERY SUCCESSFUL")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Sources used: {sources}")
        logger.info("=" * 80)
        
        return response
    
    except HTTPException as he:
        logger.error(f"HTTPException in query: {he.status_code} - {he.detail}")
        raise
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ QUERY FAILED")
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure")
async def configure_model(new_config: ModelConfig):
    """Configure the RAG system"""
    global embeddings_cache
    
    logger.info("=" * 80)
    logger.info("CONFIGURE ENDPOINT CALLED")
    logger.info(f"New config: {new_config.dict()}")
    logger.info("=" * 80)
    
    try:
        config.model = new_config.model
        logger.info(f"✓ Model set to: {config.model}")
        
        if new_config.embedding_model:
            config.embedding_model = new_config.embedding_model
            embeddings_cache = None
            logger.info(f"✓ Embedding model set to: {config.embedding_model} (cache cleared)")
        
        if new_config.chunk_size:
            config.chunk_size = new_config.chunk_size
            logger.info(f"✓ Chunk size set to: {config.chunk_size}")
        
        if new_config.chunk_overlap:
            config.chunk_overlap = new_config.chunk_overlap
            logger.info(f"✓ Chunk overlap set to: {config.chunk_overlap}")
        
        if new_config.temperature is not None:
            config.temperature = new_config.temperature
            logger.info(f"✓ Temperature set to: {config.temperature}")
        
        logger.info(f"Configuration updated: {config.to_dict()}")
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "config": config.to_dict()
        }
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=Dict[str, List[DocumentInfo]])
async def list_documents():
    """List all uploaded documents"""
    logger.info("LIST DOCUMENTS endpoint called")
    
    try:
        documents = []
        all_metadata = doc_metadata.get_all()
        logger.debug(f"Found {len(all_metadata)} documents in metadata")
        
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
                logger.debug(f"Document listed: {filename}")
            else:
                logger.warning(f"Document in metadata but not on disk: {filename}")
        
        logger.info(f"✓ Returning {len(documents)} documents")
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{filename}/preview")
async def preview_document(filename: str, max_chunks: int = 3):
    """Preview document chunks"""
    logger.info(f"PREVIEW DOCUMENT endpoint called: {filename}")
    
    try:
        chunks = vector_store.get_documents_by_source(filename)
        
        if not chunks:
            logger.error(f"Document not found: {filename}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"✓ Found {len(chunks)} chunks for {filename}")
        
        preview_chunks = chunks[:max_chunks]
        
        response = {
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
        
        logger.info(f"✓ Returning preview with {len(preview_chunks)} chunks")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document"""
    logger.info("=" * 80)
    logger.info(f"DELETE DOCUMENT endpoint called: {filename}")
    logger.info("=" * 80)
    
    try:
        file_path = UPLOAD_DIR / filename
        logger.info(f"File path: {file_path}")
        
        if not file_path.exists():
            logger.error(f"Document not found: {filename}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file
        logger.info(f"Deleting file: {file_path}")
        file_path.unlink()
        logger.info("✓ File deleted from disk")
        
        # Remove metadata
        doc_metadata.remove_document(filename)
        logger.info("✓ Metadata removed")
        
        logger.info("=" * 80)
        logger.info(f"✓ Document '{filename}' deleted successfully")
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "message": f"Document '{filename}' deleted successfully",
            "note": "Vector store rebuild recommended for complete removal"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_all():
    """Clear all documents and vector database"""
    logger.info("=" * 80)
    logger.info("CLEAR ALL endpoint called")
    logger.info("=" * 80)
    
    try:
        # Clear uploaded files
        logger.info("Clearing uploaded files...")
        file_count = 0
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
                file_count += 1
                logger.debug(f"Deleted: {file_path.name}")
        logger.info(f"✓ Deleted {file_count} files")
        
        # Clear vector data
        logger.info("Clearing vector data...")
        vector_file = DATA_DIR / 'vectors.json'
        if vector_file.exists():
            vector_file.unlink()
            logger.info("✓ Vector file deleted")
        
        # Reset
        logger.info("Resetting vector store...")
        vector_store.clear()
        logger.info("✓ Vector store cleared")
        
        logger.info("Resetting metadata...")
        doc_metadata.clear()
        logger.info("✓ Metadata cleared")
        
        logger.info("=" * 80)
        logger.info("✓ All documents and embeddings cleared successfully")
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "message": "All documents and embeddings cleared successfully"
        }
    except Exception as e:
        logger.error(f"Clear error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get detailed statistics"""
    logger.info("STATS endpoint called")
    
    try:
        stats = vector_store.get_stats()
        all_docs = doc_metadata.get_all()
        
        total_size = sum(doc.get("size", 0) for doc in all_docs.values())
        
        response = {
            "total_documents": len(all_docs),
            "total_chunks": stats.get("total_chunks", 0),
            "total_queries": stats.get("total_queries", 0),
            "total_size_bytes": total_size,
            "last_updated": stats.get("last_updated"),
            "config": config.to_dict()
        }
        
        logger.info(f"✓ Statistics: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("Starting Enhanced RAG Application Server with Debug Logging...")
    print("=" * 80)
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health")
    print(f"Log file: rag_app.log")
    print("=" * 80)
    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") UPLOAD SUCCESSFUL: {file.filename}")
        logger.info(f"Response: {response.dict()}")
        logger.info("=" * 80)
        
        return response
    
    except HTTPException as he:
        logger.error(f"HTTPException in upload: {he.status_code} - {he.detail}")
        raise
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ UPLOAD FAILED: Unexpected error")
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        
        # Clean up file if it was created
        if file_path and file_path.exists():
            logger.info(f"Cleaning up file: {file_path}")
            try:
                file_path.unlink()
                logger.info("✓ File cleaned up")
            except Exception as cleanup_error:
                logger.error(f"✗ Failed to clean up file: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    logger.info("=" * 80)
    logger.info("QUERY ENDPOINT CALLED")
    logger.info(f"Question: {request.question}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Check if documents exist
        doc_count = vector_store.get_count()
        logger.info(f"Documents in vector store: {doc_count}")
        
        if doc_count == 0:
            logger.error("No documents indexed")
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        # Get parameters
        model_name = request.model or config.model
        temperature = request.temperature if request.temperature is not None else config.temperature
        logger.info(f"Using model: {model_name}, temperature: {temperature}, top_k: {request.top_k}")
        
        # Get embeddings
        logger.info("STEP 1: Getting embeddings for query...")
        embeddings_model = get_embeddings()
        query_embedding = embeddings_model.embed_query(request.question)
        logger.info(f"✓ Query embedding generated. Dimension: {len(query_embedding)}")
        
        # Search for similar documents
        logger.info("STEP 2: Searching for similar documents...")
        similar_docs = vector_store.similarity_search(query_embedding, k=request.top_k)
        
        if not similar_docs:
            logger.error("No relevant documents found")
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found"
            )
        
        logger.info(f"✓ Found {len(similar_docs)} similar documents")
        
        # Build context
        logger.info("STEP 3: Building context...")
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in similar_docs
        ])
        logger.debug(f"Context length: {len(context)} characters")
        
        # Generate answer
        logger.info("STEP 4: Generating answer with LLM...")
        llm = Ollama(model=model_name, temperature=temperature)
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Provide a clear, concise answer based solely on the given context.

Context:
{context}

Question: {question}

Answer: """
        
        prompt = prompt_template.format(context=context, question=request.question)
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        answer = llm.invoke(prompt)
        logger.info(f"✓ Answer generated. Length: {len(answer)} characters")
        
        # Collect sources and scores
        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in similar_docs
        ]))
        
        similarity_scores = [
            doc.metadata.get("similarity_score", 0.0)
            for doc in similar_docs
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = QueryResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(similar_docs),
            similarity_scores=similarity_scores,
            processing_time=processing_time
        )
        
        logger.info("=" * 80)
        logger.info(f"✓
