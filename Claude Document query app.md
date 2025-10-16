Create an app which will follow a RAG (Retrieval-Augmented Generation) architecture with these key components:

Document Upload Interface: Allow users to upload PDF, DOCX, TXT, or other document formats.

Document Processing: Extract text, split into chunks, and generate vector embeddings.

Vector Database: Store embeddings for fast semantic search (ChromaDB or FAISS).

Query Interface: Accept user questions and retrieve relevant document chunks.

Answer Generation: Use Ollama local LLM to generate contextual answers based on retrieved content.aws.plainenglish+1​

User should be able to see the locally installed models and change the model

Technology Stack

Backend

Python 3: Core programming language.

Ollama: Run local LLMs (Llama 3, Mistral, Phi, etc.) for text generation.

Embedding Model: Use nomic-embed-text via Ollama for creating vector embeddings.

Vector Database: ChromaDB (lightweight, local) or FAISS (faster for large datasets).

LangChain: Framework to orchestrate document loading, text splitting, retrieval, and generation.

Flask/FastAPI: Web framework for API endpoints.dev+1​

Frontend (Optional)

Streamlit: Quick Python-based UI for prototyping.