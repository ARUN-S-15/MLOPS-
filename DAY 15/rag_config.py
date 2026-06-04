"""
RAG System Configuration
Handles API keys, model settings, and database configuration
"""

import os
from typing import Optional

class RAGConfig:
    """Centralized configuration for the RAG system"""
    
    # ──────────────────────── LLM CONFIGURATION ───────────────────────────
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_ID: str = os.getenv("GROQ_MODEL_ID", "mixtral-8x7b-32768")
    
    # ──────────────────────── VECTOR DATABASE ────────────────────────────
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "interview-analyzer")
    
    # ──────────────────────── EMBEDDING MODEL ────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"  # Open-source, fast
    EMBEDDING_DIMENSION: int = 384
    
    # ──────────────────────── DOCUMENT PROCESSING ────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_FORMATS: list = ["pdf", "txt", "docx", "json", "csv"]
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100
    
    # ──────────────────────── RAG PARAMETERS ───────────────────────────
    TOP_K_RETRIEVAL: int = 5  # Number of documents to retrieve
    SIMILARITY_THRESHOLD: float = 0.6
    MAX_CONTEXT_LENGTH: int = 4000  # Max tokens for context
    
    # ──────────────────────── AGENT CONFIGURATION ───────────────────────
    AGENT_TIMEOUT: int = 60  # seconds
    MAX_AGENT_ITERATIONS: int = 10
    
    # ──────────────────────── FILE STORAGE ────────────────────────────
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./cache")
    
    @classmethod
    def validate_credentials(cls) -> dict:
        """Validate that all required credentials are set"""
        missing = []
        
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
            
        return {
            "valid": len(missing) == 0,
            "missing_keys": missing
        }
    
    @classmethod
    def get_status(cls) -> dict:
        """Get configuration status"""
        return {
            "groq_configured": bool(cls.GROQ_API_KEY),
            "pinecone_configured": bool(cls.PINECONE_API_KEY),
            "embedding_model": cls.EMBEDDING_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "top_k_retrieval": cls.TOP_K_RETRIEVAL
        }
