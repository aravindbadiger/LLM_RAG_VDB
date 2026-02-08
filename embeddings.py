"""
Embedding module using Sentence Transformers.
No API key required - runs completely locally!
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

import config


class EmbeddingModel:
    """
    Wrapper for Sentence Transformer embedding models.
    Provides embeddings for text without requiring any API keys.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to config.EMBEDDING_MODEL
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {self.model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: A single string or list of strings to embed.
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            query: The query text to embed.
            
        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self.embed(query)
        return embedding[0].tolist()
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts to embed.
            
        Returns:
            List of embedding vectors (each as a list of floats).
        """
        embeddings = self.embed(documents)
        return embeddings.tolist()


# Singleton instance for reuse
_embedding_model = None


def get_embedding_model(model_name: str = None) -> EmbeddingModel:
    """
    Get or create the embedding model singleton.
    
    Args:
        model_name: Optional model name. Uses config default if not specified.
        
    Returns:
        EmbeddingModel instance.
    """
    global _embedding_model
    
    if _embedding_model is None or (model_name and model_name != _embedding_model.model_name):
        _embedding_model = EmbeddingModel(model_name)
    
    return _embedding_model


def query_embeddings(texts: List[str], model: EmbeddingModel = None) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    Compatible function signature with the lab exercises.
    
    Args:
        texts: List of text strings to embed.
        model: Optional EmbeddingModel instance. Creates one if not provided.
        
    Returns:
        List of embedding vectors.
    """
    if model is None:
        model = get_embedding_model()
    
    return model.embed_documents(texts)


if __name__ == "__main__":
    # Test the embedding model
    print("Testing Sentence Transformer embeddings...")
    
    model = get_embedding_model()
    
    # Test single text
    text = "Python is a programming language."
    embedding = model.embed_query(text)
    print(f"\nSingle text embedding:")
    print(f"  Text: {text}")
    print(f"  Embedding shape: {len(embedding)}")
    
    # Test multiple texts
    texts = [
        "Python is great for data science.",
        "Machine learning uses neural networks.",
        "Qdrant is a vector database."
    ]
    embeddings = model.embed_documents(texts)
    print(f"\nMultiple text embeddings:")
    print(f"  Number of texts: {len(texts)}")
    print(f"  Embedding shape: {len(embeddings)} x {len(embeddings[0])}")
    
    # Test similarity
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))
    
    print(f"\nSimilarity between texts:")
    for i, text_i in enumerate(texts):
        for j, text_j in enumerate(texts):
            if i < j:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                print(f"  '{text_i[:30]}...' vs '{text_j[:30]}...': {sim:.4f}")
