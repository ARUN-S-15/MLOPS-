"""
RAG Retrieval and LLM Inference Engine
Handles embedding generation, vector store operations, and LLM calls via Groq
"""

import logging
from typing import List, Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None

from rag_config import RAGConfig


class EmbeddingManager:
    """Manage embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for embeddings")
        
        self.model_name = model_name or RAGConfig.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        return [e.tolist() for e in embeddings]
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return RAGConfig.EMBEDDING_DIMENSION


class PineconeVectorStore:
    """Manage Pinecone vector database operations"""
    
    def __init__(self, api_key: str = None, environment: str = None, index_name: str = None):
        if Pinecone is None:
            raise ImportError("pinecone-client is required for Pinecone operations")
        
        self.api_key = api_key or RAGConfig.PINECONE_API_KEY
        self.environment = environment or RAGConfig.PINECONE_ENVIRONMENT
        self.index_name = index_name or RAGConfig.PINECONE_INDEX_NAME
        
        if not self.api_key:
            raise ValueError("Pinecone API key not configured")
        
        logger.info(f"Initializing Pinecone with index: {self.index_name}")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)
    
    def create_index_if_not_exists(self, dimension: int = 384):
        """Create index if it doesn't exist"""
        try:
            index_list = self.pc.list_indexes()
            index_names = [idx.name for idx in index_list]
            
            if self.index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
            else:
                logger.info(f"Index {self.index_name} already exists")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict]]):
        """
        Upsert vectors to Pinecone
        
        Args:
            vectors: List of (id, embedding, metadata) tuples
        """
        try:
            self.index.upsert(vectors=vectors)
            logger.info(f"Upserted {len(vectors)} vectors to Pinecone")
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise
    
    def query(self, query_embedding: List[float], top_k: int = 5, 
              filters: Dict = None) -> List[Dict]:
        """
        Query similar vectors from Pinecone
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters
        
        Returns:
            List of matching documents with scores
        """
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )
            
            matches = []
            for match in results.matches:
                matches.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })
            
            return matches
        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            raise
    
    def delete_vectors(self, ids: List[str]):
        """Delete vectors by ID"""
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise


class RAGRetriever:
    """Main RAG retriever combining embeddings and vector store"""
    
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: PineconeVectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
    
    def index_documents(self, chunks: List[Dict]):
        """Index document chunks into vector store"""
        
        # Extract texts and prepare for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_manager.embed_texts(texts)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{chunk['source']}_{chunk['chunk_id']}"
            metadata = {
                'content': chunk['content'],
                'source': chunk['source'],
                'doc_type': chunk['doc_type'],
                'chunk_id': chunk['chunk_id'],
                **chunk.get('metadata', {})
            }
            vectors.append((vector_id, embedding, metadata))
        
        # Upsert to Pinecone
        self.vector_store.upsert_vectors(vectors)
        logger.info(f"Indexed {len(vectors)} chunks")
    
    def retrieve_context(self, query: str, top_k: int = 5, 
                        doc_type_filter: str = None) -> List[Dict]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Query string
            top_k: Number of results
            doc_type_filter: Filter by document type (resume, job_description, etc.)
        
        Returns:
            List of relevant chunks with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Build filters if doc_type specified
        filters = None
        if doc_type_filter:
            filters = {'doc_type': {'$eq': doc_type_filter}}
        
        # Query vector store
        results = self.vector_store.query(query_embedding, top_k, filters)
        
        # Filter by similarity threshold
        threshold = RAGConfig.SIMILARITY_THRESHOLD
        results = [r for r in results if r['score'] >= threshold]
        
        return results


class GroqLLMInference:
    """Handle LLM inference via Groq API"""
    
    def __init__(self, api_key: str = None, model_id: str = None):
        if Groq is None:
            raise ImportError("groq is required for LLM inference")
        
        self.api_key = api_key or RAGConfig.GROQ_API_KEY
        self.model_id = model_id or RAGConfig.GROQ_MODEL_ID
        
        if not self.api_key:
            raise ValueError("Groq API key not configured")
        
        logger.info(f"Initializing Groq LLM: {self.model_id}")
        self.client = Groq(api_key=self.api_key)
    
    def generate(self, prompt: str, context: str = "", max_tokens: int = 1024,
                temperature: float = 0.7) -> str:
        """
        Generate response using Groq LLM
        
        Args:
            prompt: User prompt
            context: Retrieved context from vector store
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        
        Returns:
            Generated response
        """
        
        # Build system message with context
        system_message = "You are an expert AI interview coach and communication analyst."
        
        if context:
            system_message += f"\n\nRelevant Context:\n{context}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_retrieval(self, prompt: str, retriever: RAGRetriever,
                               doc_type_filter: str = None, max_tokens: int = 1024) -> Dict:
        """
        Generate response with RAG (retrieve context first)
        
        Returns:
            Dictionary with response and retrieved sources
        """
        # Retrieve context
        retrieved = retriever.retrieve_context(prompt, top_k=5, doc_type_filter=doc_type_filter)
        
        # Build context string
        context_text = "\n\n".join([
            f"[{r['metadata'].get('source', 'Unknown')}]\n{r['metadata'].get('content', '')}"
            for r in retrieved
        ])
        
        # Generate response
        response = self.generate(prompt, context_text, max_tokens)
        
        return {
            'response': response,
            'retrieved_sources': [
                {
                    'source': r['metadata'].get('source'),
                    'score': r['score'],
                    'chunk_id': r['metadata'].get('chunk_id')
                }
                for r in retrieved
            ]
        }
