"""
Qdrant vector database utilities module.
Handles collection management, data ingestion, and querying.
"""
from typing import List, Optional, Tuple
from uuid import uuid4
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Distance, VectorParams

import config
from embeddings import EmbeddingModel, get_embedding_model


class QdrantManager:
    """
    Manager class for Qdrant vector database operations.
    """
    
    def __init__(self, 
                 host: str = None, 
                 port: int = None,
                 path: str = None):
        """
        Initialize Qdrant client.
        
        Args:
            host: Qdrant server host. If None, uses local storage.
            port: Qdrant server port.
            path: Path for local Qdrant storage (if not using server).
        """
        self.host = host or config.QDRANT_HOST
        self.port = port or config.QDRANT_PORT
        
        # Use local storage if no host provided or if path is specified
        if path:
            self.client = QdrantClient(path=path)
            print(f"Connected to local Qdrant storage at: {path}")
        else:
            try:
                self.client = QdrantClient(host=self.host, port=self.port)
                # Test connection
                self.client.get_collections()
                print(f"Connected to Qdrant server at: {self.host}:{self.port}")
            except Exception as e:
                print(f"Could not connect to Qdrant server: {e}")
                print(f"Using local storage at: {config.QDRANT_STORAGE_DIR}")
                self.client = QdrantClient(path=str(config.QDRANT_STORAGE_DIR))
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return collection_name in self.list_collections()
    
    def create_collection(self, 
                          collection_name: str, 
                          embedding_dim: int = None,
                          distance: Distance = Distance.COSINE) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection to create.
            embedding_dim: Dimensionality of the vectors.
            distance: Distance metric (COSINE, EUCLID, DOT).
            
        Returns:
            True if successful, False otherwise.
        """
        embedding_dim = embedding_dim or config.EMBEDDING_DIM
        
        if self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' already exists.")
            return False
        
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=distance
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=64, 
                    ef_construct=512, 
                    on_disk=True
                )
            )
            print(f"Collection '{collection_name}' created successfully.")
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return False
        
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a collection."""
        if not self.collection_exists(collection_name):
            return {}
        
        info = self.client.get_collection(collection_name)
        # Handle API differences between qdrant-client versions
        vectors_count = getattr(info, 'vectors_count', None)
        if vectors_count is None:
            vectors_count = info.points_count  # Fallback for newer API
        return {
            'name': collection_name,
            'vectors_count': vectors_count,
            'points_count': info.points_count,
            'status': info.status.value,
        }
    
    def ingest_documents(self,
                         collection_name: str,
                         documents: List[str],
                         embedding_model: EmbeddingModel = None,
                         metadata: List[dict] = None,
                         batch_size: int = 100) -> int:
        """
        Ingest documents into a collection.
        
        Args:
            collection_name: Name of the collection.
            documents: List of document texts to ingest.
            embedding_model: Embedding model to use.
            metadata: Optional list of metadata dicts for each document.
            batch_size: Number of documents to process at once.
            
        Returns:
            Number of documents successfully ingested.
        """
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist. Creating...")
            if embedding_model:
                self.create_collection(collection_name, embedding_model.embedding_dim)
            else:
                self.create_collection(collection_name)
        
        if embedding_model is None:
            embedding_model = get_embedding_model()
        
        if metadata is None:
            metadata = [{} for _ in documents]
        
        total_ingested = 0
        
        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Ingesting"):
            batch_docs = documents[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            # Generate embeddings
            embeddings = embedding_model.embed_documents(batch_docs)
            
            # Create points
            points = []
            for doc, emb, meta in zip(batch_docs, embeddings, batch_metadata):
                point_id = int(uuid4().int >> 64)  # Truncate to 64-bit
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=emb,
                        payload={
                            "text": doc,
                            "length": len(doc),
                            **meta
                        }
                    )
                )
            
            # Upsert to database
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                total_ingested += len(points)
            except Exception as e:
                print(f"Error ingesting batch: {e}")
        
        print(f"Successfully ingested {total_ingested} documents.")
        return total_ingested
    
    def search(self,
               collection_name: str,
               query_text: str,
               embedding_model: EmbeddingModel = None,
               top_k: int = None,
               score_threshold: float = None) -> List[Tuple[str, float]]:
        """
        Search for similar documents.
        
        Args:
            collection_name: Name of the collection to search.
            query_text: Query text to search for.
            embedding_model: Embedding model to use.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.
            
        Returns:
            List of tuples (document_text, similarity_score).
        """
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return []
        
        if embedding_model is None:
            embedding_model = get_embedding_model()
        
        top_k = top_k or config.TOP_K_RESULTS
        score_threshold = score_threshold or config.SEARCH_THRESHOLD
        
        # Generate query embedding
        query_embedding = embedding_model.embed_query(query_text)
        
        # Search using query_points (qdrant-client 1.16+ API)
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        return [(r.payload.get("text", ""), r.score) for r in results.points]
    
    def search_batch(self,
                     collection_name: str,
                     query_texts: List[str],
                     embedding_model: EmbeddingModel = None,
                     top_k: int = None,
                     score_threshold: float = None) -> List[List[Tuple[str, float]]]:
        """
        Search for similar documents in batch.
        
        Args:
            collection_name: Name of the collection to search.
            query_texts: List of query texts.
            embedding_model: Embedding model to use.
            top_k: Number of results per query.
            score_threshold: Minimum similarity score.
            
        Returns:
            List of lists of tuples (document_text, similarity_score).
        """
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return []
        
        if embedding_model is None:
            embedding_model = get_embedding_model()
        
        top_k = top_k or config.TOP_K_RESULTS
        score_threshold = score_threshold or config.SEARCH_THRESHOLD
        
        # Generate query embeddings
        query_embeddings = embedding_model.embed_documents(query_texts)
        
        # Build search requests
        requests = [
            models.QueryRequest(
                query=emb,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True
            )
            for emb in query_embeddings
        ]
        
        # Search using query_batch_points (qdrant-client 1.16+ API)
        results = self.client.query_batch_points(
            collection_name=collection_name,
            requests=requests
        )
        
        return [
            [(r.payload.get("text", ""), r.score) for r in batch.points]
            for batch in results
        ]


def query_qdrant(embeddings: List[List[float]],
                 threshold: float,
                 client: QdrantClient,
                 collection: str) -> List[List[str]]:
    """
    Query Qdrant for similar documents.
    Compatible function signature with the lab exercises.
    
    Args:
        embeddings: List of embedding vectors.
        threshold: Minimum similarity score.
        client: Qdrant client instance.
        collection: Collection name.
        
    Returns:
        List of lists of matching document texts.
    """
    results = []
    
    for emb in embeddings:
        search_results = client.query_points(
            collection_name=collection,
            query=emb,
            limit=10,
            score_threshold=threshold,
            with_payload=True
        )
        
        texts = [r.payload.get("text", "") for r in search_results.points]
        results.append(texts)
    
    return results


# Singleton manager instance
_qdrant_manager = None


def get_qdrant_manager(host: str = None, 
                       port: int = None,
                       path: str = None) -> QdrantManager:
    """
    Get or create the Qdrant manager singleton.
    
    Args:
        host: Qdrant server host.
        port: Qdrant server port.
        path: Path for local storage (optional).
        
    Returns:
        QdrantManager instance.
    """
    global _qdrant_manager
    
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager(host, port, path)
    
    return _qdrant_manager


if __name__ == "__main__":
    # Test Qdrant operations
    print("Testing Qdrant utilities...")
    
    # Initialize manager (using local storage for testing)
    manager = QdrantManager(path=str(config.QDRANT_STORAGE_DIR))
    
    # List collections
    print(f"\nExisting collections: {manager.list_collections()}")
    
    # Create test collection
    test_collection = "test-collection"
    manager.delete_collection(test_collection)  # Clean up if exists
    manager.create_collection(test_collection, config.EMBEDDING_DIM)
    
    # Ingest test documents
    test_docs = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Vector databases store embeddings for semantic search.",
        "Qdrant is an open-source vector database.",
        "Neural networks are used in deep learning.",
    ]
    
    manager.ingest_documents(test_collection, test_docs)
    
    # Get collection info
    info = manager.get_collection_info(test_collection)
    print(f"\nCollection info: {info}")
    
    # Search
    query = "What is Python?"
    results = manager.search(test_collection, query)
    print(f"\nSearch results for '{query}':")
    for text, score in results:
        print(f"  {score:.4f}: {text[:50]}...")
    
    # Clean up
    manager.delete_collection(test_collection)
    print("\nTest completed.")
