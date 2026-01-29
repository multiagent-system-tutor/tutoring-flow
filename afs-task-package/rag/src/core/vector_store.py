"""
Vector store module for managing FAISS index and vector operations.

This module provides functionality for creating, saving, loading,
and managing FAISS vector stores for efficient similarity search.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..utils.logger import LoggerMixin


class VectorStoreError(Exception):
    """Custom exception for vector store errors."""
    pass


class VectorStoreManager(LoggerMixin):
    """
    Vector store manager for FAISS index operations.
    
    This class provides high-level interface for creating, saving,
    loading, and managing FAISS vector stores.
    """
    
    def __init__(
        self,
        embeddings,
        index_path: str = "faiss_index",
        save_on_create: bool = True
    ):
        """
        Initialize VectorStoreManager.
        
        Args:
            embeddings: Embedding generator instance
            index_path: Path to save/load FAISS index
            save_on_create: Automatically save index after creation
        """
        self.embeddings = embeddings
        self.index_path = Path(index_path)
        self.save_on_create = save_on_create
        self.vector_store: Optional[FAISS] = None
        
        self.logger.info(f"VectorStoreManager initialized with index_path: {index_path}")
    
    def create_index(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS index from documents.
        
        Args:
            documents: List of Document objects with embeddings
            
        Returns:
            FAISS vector store
            
        Raises:
            VectorStoreError: If index creation fails
        """
        if not documents:
            error_msg = "Cannot create index from empty document list"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
        
        self.logger.info(f"Creating FAISS index from {len(documents)} documents")
        start_time = time.time()
        
        try:
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            index_time = time.time() - start_time
            
            # Estimate memory usage (rough estimate)
            vector_dim = len(self.embeddings.embed_query("test"))
            memory_kb = (len(documents) * vector_dim * 4) / 1024  # 4 bytes per float
            
            self.logger.info(
                f"FAISS index created successfully in {index_time:.2f}s, "
                f"estimated memory: {memory_kb:.2f} KB"
            )
            
            # Save index if configured
            if self.save_on_create:
                self.save_index()
            
            return self.vector_store
            
        except Exception as e:
            error_msg = f"Failed to create FAISS index: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save FAISS index to disk.
        
        Args:
            path: Optional custom path (uses default if None)
            
        Raises:
            VectorStoreError: If saving fails
        """
        if self.vector_store is None:
            error_msg = "No vector store to save"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
        
        save_path = Path(path) if path else self.index_path
        
        self.logger.info(f"Saving FAISS index to: {save_path}")
        
        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index
            self.vector_store.save_local(str(save_path))
            
            self.logger.info("FAISS index saved successfully")
            
        except Exception as e:
            error_msg = f"Failed to save FAISS index: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def load_index(self, path: Optional[str] = None) -> FAISS:
        """
        Load FAISS index from disk.
        
        Args:
            path: Optional custom path (uses default if None)
            
        Returns:
            Loaded FAISS vector store
            
        Raises:
            VectorStoreError: If loading fails
        """
        load_path = Path(path) if path else self.index_path
        
        if not load_path.exists():
            error_msg = f"Index path does not exist: {load_path}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
        
        self.logger.info(f"Loading FAISS index from: {load_path}")
        start_time = time.time()
        
        try:
            self.vector_store = FAISS.load_local(
                str(load_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            load_time = time.time() - start_time
            
            self.logger.info(
                f"FAISS index loaded successfully in {load_time:.2f}s"
            )
            
            return self.vector_store
            
        except Exception as e:
            error_msg = f"Failed to load FAISS index: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def index_exists(self, path: Optional[str] = None) -> bool:
        """
        Check if FAISS index exists on disk.
        
        Args:
            path: Optional custom path (uses default if None)
            
        Returns:
            True if index exists
        """
        check_path = Path(path) if path else self.index_path
        exists = check_path.exists()
        
        self.logger.debug(f"Index exists at {check_path}: {exists}")
        
        return exists
    
    def get_vector_store(self) -> Optional[FAISS]:
        """
        Get the current vector store instance.
        
        Returns:
            FAISS vector store or None if not initialized
        """
        if self.vector_store is None:
            self.logger.warning("Vector store not initialized")
        
        return self.vector_store
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary containing vector store statistics
        """
        if self.vector_store is None:
            return {
                "initialized": False,
                "index_exists": self.index_exists()
            }
        
        # Get basic statistics
        stats = {
            "initialized": True,
            "index_exists": self.index_exists(),
            "index_path": str(self.index_path)
        }
        
        # Try to get vector count (this might not be available in all FAISS versions)
        try:
            if hasattr(self.vector_store, 'index'):
                stats["vector_count"] = self.vector_store.index.ntotal
        except Exception as e:
            self.logger.debug(f"Could not get vector count: {str(e)}")
        
        return stats
    
    def delete_index(self, path: Optional[str] = None) -> None:
        """
        Delete FAISS index from disk.
        
        Args:
            path: Optional custom path (uses default if None)
        """
        delete_path = Path(path) if path else self.index_path
        
        if not delete_path.exists():
            self.logger.warning(f"Index does not exist: {delete_path}")
            return
        
        try:
            import shutil
            shutil.rmtree(delete_path)
            self.logger.info(f"Deleted index at: {delete_path}")
        except Exception as e:
            self.logger.error(f"Failed to delete index: {str(e)}")
