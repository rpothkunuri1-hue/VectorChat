# rag_app.py - Fixed and Enhanced RAG Application Backend
"""
Enhanced RAG Application with bug fixes and comprehensive debugging
Requires: pip install fastapi uvicorn langchain langchain-community langchain-ollama pypdf python-docx 
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

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# LangChain imports - FIXED: Using new import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader
)
# FIXED: Updated to use langchain_ollama instead of deprecated langchain_community
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    logger.info("Using langchain_ollama (recommended)")
except ImportError:
    logger.warning("langchain_ollama not found, falling back to langchain_community")
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama as OllamaLLM

from langchain.schema import Document

# Global variables
UPLOAD_DIR = Path("./uploaded_documents")
DATA_DIR = Path("./vector_data")
METADATA_FILE = DATA_DIR / "metadata.json"

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

logger.info(f"Upload directory: {UPLOAD_DIR.absolute()}")
logger.info(f"Data directory: {DATA_DIR.absolute()}")

# Configuration
class AppConfig:
    def __init__(self):
        self.model = "llama3"
        # FIXED: Using a more reliable embedding model that won't show in LLM dropdown
        self.embedding_model = "nomic-embed-text"  # This is an embedding-specific model
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
logger.info(f"Initial config: {config.to_dict()}")

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
        logger.debug("SimpleVectorStore initialized")
        
    def add_documents(self, docs: List[Document], embeddings: List[List[float]]):
        """Add documents with their embeddings"""
        logger.debug(f"Adding {len(docs)} documents with {len(embeddings)} embeddings")
        
        # FIXED: Convert embeddings to numpy arrays with dimension validation
        try:
            new_embeddings = []
            for i, emb in enumerate(embeddings):
                if isinstance(emb, list):
                    new_emb = np.array(emb, dtype=np.float32)
                elif isinstance(emb, np.ndarray):
                    new_emb = emb.astype(np.float32)
                else:
                    logger.error(f"Unexpected embedding type at index {i}: {type(emb)}")
                    raise ValueError(f"Invalid embedding type: {type(emb)}")
                
                # Check dimension consistency
                if self.embeddings_list and len(new_emb) != len(self.embeddings_list[0]):
                    logger.error(f"Embedding dimension mismatch at index {i}!")
                    logger.error(f"Expected dimension: {len(self.embeddings_list[0])}, Got: {len(new_emb)}")
                    logger.error(f"Current embedding model: {config.embedding_model}")
                    raise ValueError(
                        f"Embedding dimension mismatch! Expected {len(self.embeddings_list[0])}, got {len(new_emb)}. "
                        f"This usually means documents were embedded with different models. "
                        f"Please clear all documents and re-upload them with the current embedding model."
                    )
                
                new_embeddings.append(new_emb)
            
            self.embeddings_list.extend(new_embeddings)
            self.documents.extend(docs)
            self.metadata.extend([doc.metadata for doc in docs])
            self.stats["total_chunks"] = len(self.documents)
            self.stats["last_updated"] = datetime.now().isoformat()
            
            logger.info(f"Successfully added {len(docs)} documents. Total chunks: {self.stats['total_chunks']}")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            logger.error(traceback.format_exc())
            raise
        
    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> List[Document]:
        """Find most similar documents"""
        logger.debug(f"Performing similarity search with k={k}")
        
        if not self.embeddings_list:
            logger.warning("No embeddings available for search")
            return []
        
        try:
            # Ensure query_embedding is the right shape
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            query_embedding = query_embedding.reshape(1, -1)
            
            # Stack all embeddings for efficient computation
            embeddings_matrix = np.vstack(self.embeddings_list)
            
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
            
            logger.debug(f"Computed similarities: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
            
            top_k = min(k, len(similarities))
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for i in top_indices:
                doc = self.documents[i]
                doc.metadata["similarity_score"] = float(similarities[i])
                results.append(doc)
                logger.debug(f"Result {len(results)}: source={doc.metadata.get('source')}, score={similarities[i]:.4f}")
            
            self.stats["total_queries"] += 1
            logger.info(f"Similarity search completed. Returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_count(self) -> int:
        count = len(self.documents)
        logger.debug(f"Current document count: {count}")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()
    
    def get_documents_by_source(self, source: str) -> List[Document]:
        """Get all documents from a specific source"""
        docs = [doc for doc in self.documents if doc.metadata.get("source") == source]
        logger.debug(f"Found {len(docs)} documents for source: {source}")
        return docs
    
    def save(self):
        """Save to disk"""
        try:
            logger.debug("Saving vector store to disk...")
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
            
            vector_file = DATA_DIR / 'vectors.json'
            with open(vector_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.documents)} documents to {vector_file}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def load(self) -> bool:
        """Load from disk"""
        try:
            vector_file = DATA_DIR / 'vectors.json'
            if not vector_file.exists():
                logger.info("No existing vector store found")
                return False
            
            logger.debug(f"Loading vector store from {vector_file}")
            with open(vector_file, 'r') as f:
                data = json.load(f)
            
            self.embeddings_list = [np.array(emb, dtype=np.float32) for emb in data['embeddings']]
            self.documents = [
                Document(page_content=doc['page_content'], metadata=doc['metadata'])
                for doc in data['documents']
            ]
            self.metadata = data['metadata']
            self.stats = data.get('stats', self.stats)
            
            # Validate embedding dimensions
            if self.embeddings_list:
                dimensions = [len(emb) for emb in self.embeddings_list]
                unique_dims = set(dimensions)
                if len(unique_dims) > 1:
                    logger.error(f"Loaded vector store has inconsistent embedding dimensions: {unique_dims}")
                    logger.error("This indicates documents were embedded with different models.")
                    logger.error("Clearing corrupted vector store...")
                    self.clear()
                    return False
                logger.info(f"Loaded embeddings with dimension: {dimensions[0]}")
            
            logger.info(f"Loaded {len(self.documents)} documents from disk")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def clear(self):
        """Clear all data"""
        logger.info("Clearing vector store")
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
        logger.debug(f"DocumentMetadataManager initialized with {len(self.metadata)} documents")
    
    def add_document(self, filename: str, chunks: int, file_size: int, file_type: str):
        logger.debug(f"Adding metadata for {filename}: {chunks} chunks, {file_size} bytes")
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
            logger.info(f"Removing metadata for {filename}")
            del self.metadata[filename]
            self.save()
    
    def get_document(self, filename: str) -> Optional[Dict]:
        return self.metadata.get(filename)
    
    def get_all(self) -> Dict:
        return self.metadata.copy()
    
    def clear(self):
        logger.info("Clearing all metadata")
        self.metadata = {}
        self.save()
    
    def save(self):
        try:
            with open(METADATA_FILE, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"Metadata saved to {METADATA_FILE}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def load(self):
        try:
            if METADATA_FILE.exists():
                with open(METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)
                logger.debug(f"Loaded metadata for {len(self.metadata)} documents")
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
    logger.info("=" * 80)
    logger.info("Starting RAG Application Server...")
    logger.info("=" * 80)
    vector_store.load()
    logger.info(f"Loaded {vector_store.get_count()} document chunks")
    logger.info("Server ready to accept requests")
    logger.info("=" * 80)
    yield
    # Shutdown
    logger.info("=" * 80)
    logger.info("Shutting down RAG Application Server...")
    logger.info("=" * 80)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Application API",
    version="2.1.0",
    description="Enhanced RAG Application with Fixed Vector Store",
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
            logger.debug(f"Initializing embeddings with model: {config.embedding_model}")
            embeddings_cache = OllamaEmbeddings(
                model=config.embedding_model,
                base_url="http://localhost:11434"
            )
            # Test the embeddings
            test_result = embeddings_cache.embed_query("test")
            logger.info(f"Embeddings initialized successfully. Dimension: {len(test_result)}")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to initialize embeddings: {str(e)}. Make sure Ollama is running and the model '{config.embedding_model}' is available.")
    return embeddings_cache

def load_document(file_path: str) -> List[Document]:
    """Load document based on file extension"""
    ext = Path(file_path).suffix.lower()
    logger.debug(f"Loading document: {file_path} (type: {ext})")
    
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
        
        # Debug: Show first document preview
        if documents:
            preview = documents[0].page_content[:200]
            logger.debug(f"First page preview: {preview}...")
        
        return documents
    except Exception as e:
        logger.error(f"Failed to load document {file_path}: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to load document: {str(e)}")

def process_document(file_path: str, filename: str) -> int:
    """Process document: load, split, embed, and store"""
    logger.info(f"Processing document: {filename}")
    try:
        # Load document
        documents = load_document(file_path)
        logger.debug(f"Loaded {len(documents)} pages")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = filename
            chunk.metadata["chunk_id"] = i
            if i == 0:
                logger.debug(f"Sample chunk {i}: length={len(chunk.page_content)}, content={chunk.page_content[:100]}...")
        
        # Generate embeddings
        embeddings_model = get_embeddings()
        texts = [chunk.page_content for chunk in chunks]
        
        logger.debug(f"Generating embeddings for {len(texts)} chunks...")
        chunk_embeddings = embeddings_model.embed_documents(texts)
        logger.info(f"Generated {len(chunk_embeddings)} embeddings")
        
        # Verify embeddings structure
        if chunk_embeddings:
            first_emb = chunk_embeddings[0]
            logger.debug(f"First embedding type: {type(first_emb)}, length: {len(first_emb) if hasattr(first_emb, '__len__') else 'N/A'}")
        
        # Add to vector store
        vector_store.add_documents(chunks, chunk_embeddings)
        vector_store.save()
        
        logger.info(f"Successfully processed {filename}: {len(chunks)} chunks created and saved")
        return len(chunks)
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        logger.error(traceback.format_exc())
        raise

# API Endpoints

@app.get("/")
async def root():
    logger.debug("Root endpoint accessed")
    return {
        "message": "RAG Application API - Fixed Version",
        "version": "2.1.0",
        "status": "running",
        "database": "In-Memory + File Storage",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint accessed")
    stats = vector_store.get_stats()
    
    # Check if Ollama is accessible
    ollama_status = "unknown"
    try:
        embeddings_model = get_embeddings()
        test_embedding = embeddings_model.embed_query("test")
        ollama_status = "healthy"
        logger.debug(f"Ollama health check passed. Embedding dimension: {len(test_embedding)}")
    except Exception as e:
        ollama_status = f"error: {str(e)}"
        logger.warning(f"Ollama health check failed: {e}")
    
    response = {
        "status": "healthy",
        "vector_db": "SimpleVectorStore",
        "documents_indexed": vector_store.get_count(),
        "total_queries": stats.get("total_queries", 0),
        "last_updated": stats.get("last_updated"),
        "ollama_status": ollama_status,
        "config": config.to_dict()
    }
    logger.info(f"Health check response: {response}")
    return response

@app.get("/models")
async def get_available_models():
    """Get list of available Ollama models - FIXED to filter out embedding models"""
    logger.debug("Fetching available models")
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
            all_models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    all_models.append(model_name)
            
            # FIXED: Filter out embedding-only models from the LLM dropdown
            embedding_keywords = ['embed', 'nomic', 'bge', 'e5']
            llm_models = [
                m for m in all_models 
                if not any(keyword in m.lower() for keyword in embedding_keywords)
            ]
            
            logger.info(f"Found {len(all_models)} total models, {len(llm_models)} LLM models")
            logger.debug(f"All models: {all_models}")
            logger.debug(f"LLM models: {llm_models}")
            
            return {
                "models": llm_models if llm_models else ["llama3", "mistral", "phi"],
                "current": config.model,
                "embedding_model": config.embedding_model
            }
        else:
            logger.warning(f"Ollama list command failed: {result.stderr}")
            return {
                "models": ["llama3", "mistral", "phi", "gemma"],
                "current": config.model,
                "embedding_model": config.embedding_model
            }
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        logger.error(traceback.format_exc())
        return {
            "models": ["llama3", "mistral", "phi", "gemma"],
            "current": config.model,
            "embedding_model": config.embedding_model,
            "error": str(e)
        }

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document - FIXED"""
    logger.info(f"Upload request received for file: {file.filename}")
    file_path = None
    
    try:
        # Validate file extension
        allowed_extensions = ['.pdf', '.txt', '.docx', '.doc']
        file_ext = Path(file.filename).suffix.lower()
        
        logger.debug(f"File extension: {file_ext}")
        
        if file_ext not in allowed_extensions:
            logger.warning(f"Unsupported file type: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        file_path = UPLOAD_DIR / file.filename
        logger.debug(f"Target file path: {file_path}")
        
        # Check if file already exists
        if file_path.exists():
            logger.warning(f"File already exists: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' already exists. Please delete it first or rename the file."
            )
        
        # Save file
        logger.debug("Reading file content...")
        content = await file.read()
        file_size = len(content)
        logger.info(f"File size: {file_size} bytes")
        
        logger.debug(f"Writing file to {file_path}...")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        logger.info(f"File saved successfully to {file_path}")
        
        # Process document
        logger.debug("Starting document processing...")
        num_chunks = process_document(str(file_path), file.filename)
        logger.info(f"Document processing complete: {num_chunks} chunks created")
        
        # Update metadata
        logger.debug("Updating document metadata...")
        doc_metadata.add_document(
            filename=file.filename,
            chunks=num_chunks,
            file_size=file_size,
            file_type=file_ext
        )
        
        response = DocumentUploadResponse(
            status="success",
            filename=file.filename,
            chunks=num_chunks,
            file_size=file_size,
            message=f"Document processed successfully with {num_chunks} chunks"
        )
        logger.info(f"Upload successful: {response}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error for {file.filename}: {e}")
        logger.error(traceback.format_exc())
        
        # Clean up file if it was created
        if file_path and file_path.exists():
            logger.debug(f"Cleaning up failed upload: {file_path}")
            try:
                file_path.unlink()
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up file: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    start_time = datetime.now()
    logger.info(f"Query request received: '{request.question[:100]}...'")
    logger.debug(f"Query params: model={request.model}, top_k={request.top_k}, temperature={request.temperature}")
    
    try:
        # Check if documents are indexed
        doc_count = vector_store.get_count()
        logger.debug(f"Documents in vector store: {doc_count}")
        
        if doc_count == 0:
            logger.warning("No documents indexed")
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        # Determine model and temperature
        model_name = request.model or config.model
        temperature = request.temperature if request.temperature is not None else config.temperature
        logger.debug(f"Using model: {model_name}, temperature: {temperature}")
        
        # Get embeddings model
        logger.debug("Getting embeddings model...")
        embeddings_model = get_embeddings()
        
        # Generate query embedding
        logger.debug("Generating query embedding...")
        query_embedding = embeddings_model.embed_query(request.question)
        logger.debug(f"Query embedding generated: dimension={len(query_embedding)}")
        
        # Search for similar documents
        logger.debug(f"Searching for top {request.top_k} similar documents...")
        similar_docs = vector_store.similarity_search(query_embedding, k=request.top_k)
        
        if not similar_docs:
            logger.warning("No relevant documents found")
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found"
            )
        
        logger.info(f"Found {len(similar_docs)} relevant documents")
        
        # Build context
        context_parts = []
        for i, doc in enumerate(similar_docs):
            source = doc.metadata.get('source', 'Unknown')
            score = doc.metadata.get('similarity_score', 0.0)
            context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{doc.page_content}")
            logger.debug(f"Context chunk {i+1}: source={source}, score={score:.4f}, length={len(doc.page_content)}")
        
        context = "\n\n".join(context_parts)
        logger.debug(f"Total context length: {len(context)} characters")
        
        # Initialize LLM
        logger.debug(f"Initializing LLM: {model_name}")
        llm = OllamaLLM(model=model_name, temperature=temperature)
        
        # Create prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Provide a clear, concise answer based solely on the given context.

Context:
{context}

Question: {question}

Answer: """
        
        prompt = prompt_template.format(context=context, question=request.question)
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Generate answer
        logger.debug("Generating answer from LLM...")
        answer = llm.invoke(prompt)
        logger.info(f"Answer generated: {len(answer)} characters")
        logger.debug(f"Answer preview: {answer[:200]}...")
        
        # Extract sources
        sources = list(set([
            doc.metadata.get("source", "Unknown")
            for doc in similar_docs
        ]))
        logger.debug(f"Sources: {sources}")
        
        # Extract similarity scores
        similarity_scores = [
            doc.metadata.get("similarity_score", 0.0)
            for doc in similar_docs
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        response = QueryResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(similar_docs),
            similarity_scores=similarity_scores,
            processing_time=processing_time
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/configure")
async def configure_model(new_config: ModelConfig):
    """Configure the RAG system"""
    global embeddings_cache
    
    logger.info(f"Configuration update requested: {new_config}")
    
    try:
        old_config = config.to_dict()
        embedding_model_changed = False
        
        config.model = new_config.model
        logger.debug(f"Model updated: {new_config.model}")
        
        if new_config.embedding_model:
            if new_config.embedding_model != config.embedding_model:
                logger.warning(f"Embedding model changed from {config.embedding_model} to {new_config.embedding_model}")
                logger.warning("Changing embedding model will make existing embeddings incompatible!")
                config.embedding_model = new_config.embedding_model
                embeddings_cache = None  # Reset cache
                embedding_model_changed = True
        
        if new_config.chunk_size:
            logger.debug(f"Chunk size updated: {new_config.chunk_size}")
            config.chunk_size = new_config.chunk_size
        
        if new_config.chunk_overlap:
            logger.debug(f"Chunk overlap updated: {new_config.chunk_overlap}")
            config.chunk_overlap = new_config.chunk_overlap
        
        if new_config.temperature is not None:
            logger.debug(f"Temperature updated: {new_config.temperature}")
            config.temperature = new_config.temperature
        
        new_config_dict = config.to_dict()
        logger.info(f"Configuration updated successfully")
        logger.debug(f"Old config: {old_config}")
        logger.debug(f"New config: {new_config_dict}")
        
        response = {
            "status": "success",
            "config": new_config_dict,
            "changed_fields": [k for k in new_config_dict.keys() if new_config_dict[k] != old_config.get(k)]
        }
        
        if embedding_model_changed:
            response["warning"] = "Embedding model changed. Please clear all documents and re-upload to avoid dimension mismatches."
        
        return response
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@app.get("/documents", response_model=Dict[str, List[DocumentInfo]])
async def list_documents():
    """List all uploaded documents"""
    logger.debug("Listing all documents")
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
            logger.debug(f"Document: {filename}, chunks={meta.get('chunks', 0)}, size={meta.get('size', 0)}")
        else:
            logger.warning(f"Document {filename} in metadata but file not found")
    
    logger.info(f"Returning {len(documents)} documents")
    return {"documents": documents}

@app.get("/documents/{filename}/preview")
async def preview_document(filename: str, max_chunks: int = 3):
    """Preview document chunks"""
    logger.info(f"Preview request for document: {filename}")
    logger.debug(f"Max chunks to preview: {max_chunks}")
    
    try:
        chunks = vector_store.get_documents_by_source(filename)
        
        if not chunks:
            logger.warning(f"Document not found in vector store: {filename}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Found {len(chunks)} chunks for {filename}")
        preview_chunks = chunks[:max_chunks]
        
        preview_data = []
        for i, chunk in enumerate(preview_chunks):
            chunk_preview = {
                "chunk_id": chunk.metadata.get("chunk_id", i),
                "content": chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
                "length": len(chunk.page_content)
            }
            preview_data.append(chunk_preview)
            logger.debug(f"Preview chunk {i}: id={chunk_preview['chunk_id']}, length={chunk_preview['length']}")
        
        response = {
            "filename": filename,
            "total_chunks": len(chunks),
            "preview": preview_data
        }
        
        logger.info(f"Returning preview with {len(preview_data)} chunks")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview error for {filename}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document"""
    logger.info(f"Delete request for document: {filename}")
    
    try:
        file_path = UPLOAD_DIR / filename
        
        if not file_path.exists():
            logger.warning(f"Document file not found: {filename}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file
        logger.debug(f"Deleting file: {file_path}")
        file_path.unlink()
        logger.info(f"File deleted: {filename}")
        
        # Remove metadata
        doc_metadata.remove_document(filename)
        logger.debug(f"Metadata removed for: {filename}")
        
        # Note: Not removing from vector store to avoid rebuilding
        # This is a design choice for performance
        
        response = {
            "status": "success",
            "message": f"Document '{filename}' deleted successfully",
            "note": "Vector store rebuild recommended for complete removal"
        }
        logger.info(f"Delete successful: {filename}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error for {filename}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@app.delete("/clear")
async def clear_all():
    """Clear all documents and vector database"""
    logger.info("Clear all request received")
    
    try:
        # Count before clearing
        file_count = len(list(UPLOAD_DIR.iterdir()))
        doc_count = vector_store.get_count()
        logger.debug(f"Before clear: {file_count} files, {doc_count} document chunks")
        
        # Clear uploaded files
        logger.debug("Clearing uploaded files...")
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
                logger.debug(f"Deleted file: {file_path.name}")
        
        # Clear vector data
        vector_file = DATA_DIR / 'vectors.json'
        if vector_file.exists():
            logger.debug(f"Deleting vector file: {vector_file}")
            vector_file.unlink()
        
        # Reset in-memory structures
        logger.debug("Resetting vector store...")
        vector_store.clear()
        
        logger.debug("Clearing metadata...")
        doc_metadata.clear()
        
        logger.info("All documents and embeddings cleared successfully")
        
        return {
            "status": "success",
            "message": "All documents and embeddings cleared successfully",
            "cleared": {
                "files": file_count,
                "chunks": doc_count
            }
        }
    except Exception as e:
        logger.error(f"Clear error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@app.post("/rebuild-vectors")
async def rebuild_vectors():
    """Rebuild vector store from existing documents with current embedding model"""
    logger.info("Rebuild vectors request received")
    
    try:
        # Get all uploaded files
        uploaded_files = [f for f in UPLOAD_DIR.iterdir() if f.is_file()]
        
        if not uploaded_files:
            logger.warning("No documents to rebuild")
            raise HTTPException(status_code=400, detail="No documents found to rebuild")
        
        logger.info(f"Rebuilding vectors for {len(uploaded_files)} documents")
        
        # Clear existing vector store
        logger.debug("Clearing existing vector store...")
        vector_store.clear()
        
        # Clear vector file
        vector_file = DATA_DIR / 'vectors.json'
        if vector_file.exists():
            vector_file.unlink()
        
        # Reprocess each document
        results = []
        errors = []
        
        for file_path in uploaded_files:
            try:
                logger.info(f"Reprocessing: {file_path.name}")
                
                # Get file info
                file_size = file_path.stat().st_size
                file_ext = file_path.suffix.lower()
                
                # Process document
                num_chunks = process_document(str(file_path), file_path.name)
                
                # Update metadata
                doc_metadata.add_document(
                    filename=file_path.name,
                    chunks=num_chunks,
                    file_size=file_size,
                    file_type=file_ext
                )
                
                results.append({
                    "filename": file_path.name,
                    "chunks": num_chunks,
                    "status": "success"
                })
                logger.info(f"Successfully rebuilt: {file_path.name} ({num_chunks} chunks)")
                
            except Exception as e:
                logger.error(f"Failed to rebuild {file_path.name}: {e}")
                errors.append({
                    "filename": file_path.name,
                    "error": str(e)
                })
        
        response = {
            "status": "success" if not errors else "partial_success",
            "message": f"Rebuilt {len(results)} documents with current embedding model",
            "embedding_model": config.embedding_model,
            "successful": results,
            "failed": errors,
            "total_chunks": vector_store.get_count()
        }
        
        logger.info(f"Rebuild complete: {len(results)} success, {len(errors)} failed")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get detailed statistics"""
    logger.debug("Statistics request received")
    
    try:
        stats = vector_store.get_stats()
        all_docs = doc_metadata.get_all()
        
        total_size = sum(doc.get("size", 0) for doc in all_docs.values())
        
        # Calculate additional stats
        avg_chunks_per_doc = stats.get("total_chunks", 0) / len(all_docs) if all_docs else 0
        
        response = {
            "total_documents": len(all_docs),
            "total_chunks": stats.get("total_chunks", 0),
            "total_queries": stats.get("total_queries", 0),
            "total_size_bytes": total_size,
            "avg_chunks_per_document": round(avg_chunks_per_doc, 2),
            "last_updated": stats.get("last_updated"),
            "config": config.to_dict()
        }
        
        logger.info(f"Statistics: {response}")
        return response
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

@app.get("/debug/embeddings")
async def debug_embeddings():
    """Debug endpoint to check embeddings configuration"""
    logger.debug("Debug embeddings endpoint accessed")
    
    try:
        embeddings_model = get_embeddings()
        
        # Test embedding
        test_text = "This is a test sentence."
        logger.debug(f"Testing embedding with: '{test_text}'")
        
        test_embedding = embeddings_model.embed_query(test_text)
        
        info = {
            "status": "success",
            "embedding_model": config.embedding_model,
            "test_embedding_dimension": len(test_embedding),
            "test_embedding_type": str(type(test_embedding)),
            "first_5_values": test_embedding[:5] if len(test_embedding) >= 5 else test_embedding,
            "vector_store_embeddings_count": len(vector_store.embeddings_list),
            "vector_store_documents_count": len(vector_store.documents)
        }
        
        logger.info(f"Embeddings debug info: {info}")
        return info
    except Exception as e:
        logger.error(f"Debug embeddings error: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/vector-store")
async def debug_vector_store():
    """Debug endpoint to check vector store state"""
    logger.debug("Debug vector store endpoint accessed")
    
    try:
        stats = vector_store.get_stats()
        
        # Check embedding dimensions
        embedding_dimensions = []
        if vector_store.embeddings_list:
            embedding_dimensions = [len(emb) for emb in vector_store.embeddings_list]
            unique_dims = list(set(embedding_dimensions))
        else:
            unique_dims = []
        
        # Sample data
        sample_docs = []
        for i, doc in enumerate(vector_store.documents[:3]):
            sample_docs.append({
                "index": i,
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content_preview": doc.page_content[:100] + "...",
                "content_length": len(doc.page_content),
                "embedding_dimension": len(vector_store.embeddings_list[i]) if i < len(vector_store.embeddings_list) else None
            })
        
        info = {
            "status": "success",
            "total_documents": len(vector_store.documents),
            "total_embeddings": len(vector_store.embeddings_list),
            "embeddings_match": len(vector_store.documents) == len(vector_store.embeddings_list),
            "unique_embedding_dimensions": unique_dims,
            "dimension_consistency": len(unique_dims) <= 1,
            "stats": stats,
            "sample_documents": sample_docs
        }
        
        if vector_store.embeddings_list:
            info["embedding_dimension"] = len(vector_store.embeddings_list[0])
            info["embedding_dtype"] = str(vector_store.embeddings_list[0].dtype)
        
        # Warning if dimensions are inconsistent
        if len(unique_dims) > 1:
            info["warning"] = f"Inconsistent embedding dimensions detected: {unique_dims}. Please clear all documents and re-upload."
        
        logger.info(f"Vector store debug info: {info}")
        return info
    except Exception as e:
        logger.error(f"Debug vector store error: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Custom exception handler for better error logging
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    logger.error(traceback.format_exc())
    return {
        "detail": str(exc),
        "path": str(request.url.path),
        "type": type(exc).__name__
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("Starting Enhanced RAG Application Server...")
    print("=" * 80)
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print(f"📊 Statistics: http://localhost:8000/stats")
    print(f"🔧 Debug Embeddings: http://localhost:8000/debug/embeddings")
    print(f"🔍 Debug Vector Store: http://localhost:8000/debug/vector-store")
    print(f"🔄 Rebuild Vectors: POST http://localhost:8000/rebuild-vectors")
    print("=" * 80)
    print(f"📁 Upload Directory: {UPLOAD_DIR.absolute()}")
    print(f"💾 Data Directory: {DATA_DIR.absolute()}")
    print("=" * 80)
    print("💡 TIP: If you get embedding dimension errors, use /rebuild-vectors")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
