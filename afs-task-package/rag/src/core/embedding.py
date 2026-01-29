"""
Embedding generation module for creating vector representations of text.

This module provides functionality for generating embeddings using
HuggingFace models with proper error handling and performance tracking.
"""

import time
from typing import List, Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

from ..utils.logger import LoggerMixin


class EmbeddingError(Exception):
    """Custom exception for embedding errors."""
    pass


class EmbeddingGenerator(LoggerMixin):
    """
    Embedding generator for creating vector representations of text.
    
    This class wraps HuggingFace embedding models and provides
    functionality for batch embedding generation with error handling.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize EmbeddingGenerator.
        
        Args:
            model_name: Name of the HuggingFace embedding model
            device: Device for inference ('cpu' or 'cuda')
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        self.logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        
        try:
            start_time = time.time()
            
            # Initialize HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': normalize_embeddings}
            )
            
            load_time = time.time() - start_time
            
            self.logger.info(
                f"Embedding model loaded successfully in {load_time:.2f} seconds"
            )
            
            # Test embedding to get dimension
            test_embedding = self.embed_query("test")
            self.dimension = len(test_embedding)
            self.logger.info(f"Embedding dimension: {self.dimension}")
            
        except Exception as e:
            error_msg = f"Failed to initialize embedding model: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            self.logger.warning("Empty text list provided for embedding")
            return []
        
        self.logger.debug(f"Generating embeddings for {len(texts)} documents")
        start_time = time.time()
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            
            embed_time = time.time() - start_time
            avg_time = embed_time / len(texts) if texts else 0
            
            self.logger.info(
                f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s "
                f"({avg_time*1000:.2f}ms per doc)"
            )
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for query embedding")
            return [0.0] * self.dimension if hasattr(self, 'dimension') else []
        
        self.logger.debug(f"Generating query embedding for text: {text[:50]}...")
        start_time = time.time()
        
        try:
            embedding = self.embeddings.embed_query(text)
            
            embed_time = time.time() - start_time
            
            self.logger.debug(
                f"Generated query embedding in {embed_time*1000:.2f}ms"
            )
            
            return embedding
            
        except Exception as e:
            error_msg = f"Error generating query embedding: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    def test_embedding(self, text: str = "Test embedding generation") -> dict:
        """
        Test embedding generation and return performance metrics.
        
        Args:
            text: Test text
            
        Returns:
            Dictionary containing test results and metrics
        """
        self.logger.info("Running embedding test")
        
        try:
            start_time = time.time()
            embedding = self.embed_query(text)
            inference_time = time.time() - start_time
            
            results = {
                "success": True,
                "dimension": len(embedding),
                "inference_time_ms": inference_time * 1000,
                "sample_values": embedding[:5] if len(embedding) >= 5 else embedding
            }
            
            self.logger.info(
                f"Embedding test successful: dim={results['dimension']}, "
                f"time={results['inference_time_ms']:.2f}ms"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Embedding test failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension if hasattr(self, 'dimension') else None,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size
        }
