"""
RAG Service module providing high-level API for Teacher Agent.

This module provides a clean, production-ready interface for the
complete RAG pipeline, integrating all core components.
"""

import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core import (
    DataLoader,
    TextPreprocessor,
    DocumentChunker,
    EmbeddingGenerator,
    VectorStoreManager,
    Retriever
)
from ..config import Settings
from ..utils.logger import LoggerMixin


class RAGServiceError(Exception):
    """Custom exception for RAG service errors."""
    pass


class RAGService(LoggerMixin):
    """
    Production-ready RAG Service for Teacher Agent.
    
    This class provides a high-level interface for the complete RAG pipeline,
    handling document ingestion, processing, indexing, and retrieval with
    proper error handling and logging.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize RAG Service.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.initialized = False
        
        # Components (initialized in build_pipeline)
        self.data_loader: Optional[DataLoader] = None
        self.preprocessor: Optional[TextPreprocessor] = None
        self.chunker: Optional[DocumentChunker] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.retriever: Optional[Retriever] = None
        
        # Metrics
        self.metrics = {
            "total_documents": 0,
            "total_chunks": 0,
            "pipeline_build_time": 0.0
        }
        
        self.logger.info("RAGService initialized")
    
    def build_pipeline(self, force_rebuild: bool = False) -> None:
        """
        Build complete RAG pipeline.
        
        This method orchestrates the entire pipeline:
        1. Load documents
        2. Preprocess text
        3. Chunk documents
        4. Generate embeddings and create vector store
        5. Initialize retriever
        
        Args:
            force_rebuild: Force rebuild even if index exists
            
        Raises:
            RAGServiceError: If pipeline build fails
        """
        self.logger.info("="*60)
        self.logger.info("BUILDING RAG PIPELINE")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Initialize embedding generator first
            self._initialize_embedding_generator()
            
            # Check if we can load existing index
            self.vector_store_manager = VectorStoreManager(
                embeddings=self.embedding_generator.embeddings,
                index_path=self.settings.vector_store.index_path,
                save_on_create=self.settings.vector_store.save_on_create
            )
            
            if self.vector_store_manager.index_exists() and not force_rebuild:
                self.logger.info("Existing index found, loading...")
                self._load_existing_pipeline()
            else:
                self.logger.info("Building pipeline from scratch...")
                self._build_from_scratch()
            
            # Initialize retriever
            self._initialize_retriever()
            
            self.metrics["pipeline_build_time"] = time.time() - start_time
            self.initialized = True
            
            self.logger.info("="*60)
            self.logger.info("PIPELINE BUILD COMPLETE")
            self.logger.info(f"Total time: {self.metrics['pipeline_build_time']:.2f}s")
            self.logger.info("="*60)
            
            self._log_metrics()
            
        except Exception as e:
            error_msg = f"Failed to build pipeline: {str(e)}"
            self.logger.error(error_msg)
            raise RAGServiceError(error_msg) from e
    
    def _initialize_embedding_generator(self) -> None:
        """Initialize embedding generator."""
        self.logger.info("Initializing embedding generator...")
        
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.settings.embedding.model_name,
            device=self.settings.embedding.device,
            normalize_embeddings=self.settings.embedding.normalize_embeddings,
            batch_size=self.settings.embedding.batch_size
        )
    
    def _build_from_scratch(self) -> None:
        """Build pipeline from scratch."""
        # Step 1: Load documents
        self.logger.info("Step 1: Loading documents...")
        self.data_loader = DataLoader(
            data_dir=self.settings.data.data_dir,
            supported_formats=self.settings.data.supported_formats,
            encoding=self.settings.data.encoding
        )
        documents = self.data_loader.load_documents()
        self.metrics["total_documents"] = len(documents)
        
        if not documents:
            raise RAGServiceError("No documents loaded")
        
        # Step 2: Preprocess
        self.logger.info("Step 2: Preprocessing...")
        self.preprocessor = TextPreprocessor(
            remove_special_chars=self.settings.preprocessing.remove_special_chars,
            normalize_whitespace=self.settings.preprocessing.normalize_whitespace,
            preserve_case=self.settings.preprocessing.preserve_case
        )
        processed_docs = self.preprocessor.preprocess_documents(documents)
        
        # Step 3: Chunk
        self.logger.info("Step 3: Chunking...")
        self.chunker = DocumentChunker(
            chunk_size=self.settings.chunking.chunk_size,
            chunk_overlap=self.settings.chunking.chunk_overlap,
            separators=self.settings.chunking.separators
        )
        chunks = self.chunker.chunk_documents(processed_docs)
        self.metrics["total_chunks"] = len(chunks)
        
        # Step 4: Create vector store
        self.logger.info("Step 4: Creating vector store...")
        self.vector_store_manager.create_index(chunks)
    
    def _load_existing_pipeline(self) -> None:
        """Load existing pipeline from saved index."""
        self.vector_store_manager.load_index()
        self.logger.info("Pipeline loaded from existing index")
    
    def _initialize_retriever(self) -> None:
        """Initialize retriever."""
        self.logger.info("Initializing retriever...")
        
        vector_store = self.vector_store_manager.get_vector_store()
        
        if vector_store is None:
            raise RAGServiceError("Vector store not initialized")
        
        self.retriever = Retriever(
            vector_store=vector_store,
            top_k=self.settings.retrieval.top_k,
            score_threshold=self.settings.retrieval.score_threshold,
            return_scores=self.settings.retrieval.return_scores
        )
    
    def query(
        self,
        student_input: str,
        summary: str = "",
        student_profile: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process student query and retrieve relevant context.
        
        Args:
            student_input: Student's answer or question
            summary: Summary of student's work
            student_profile: Student profile information
            top_k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            Dictionary containing context and metadata
            
        Raises:
            RAGServiceError: If query processing fails
        """
        if not self.initialized:
            raise RAGServiceError("Pipeline not initialized. Call build_pipeline() first.")
        
        self.logger.info("Processing student query")
        
        try:
            result = self.retriever.process_query(
                student_input=student_input,
                summary=summary,
                student_profile=student_profile,
                k=top_k
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            self.logger.error(error_msg)
            raise RAGServiceError(error_msg) from e
    
    def batch_query(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query dictionaries with keys:
                    - student_input (required)
                    - summary (optional)
                    - student_profile (optional)
                    - top_k (optional)
        
        Returns:
            List of result dictionaries
        """
        self.logger.info(f"Processing {len(queries)} queries in batch")
        
        results = []
        for i, query_data in enumerate(queries):
            try:
                result = self.query(**query_data)
                results.append(result)
                self.logger.debug(f"Processed query {i+1}/{len(queries)}")
            except Exception as e:
                self.logger.error(f"Failed to process query {i+1}: {str(e)}")
                results.append({"error": str(e)})
        
        return results
    
    def rebuild_index(self) -> None:
        """
        Rebuild the vector index from scratch.
        
        Useful when documents have been updated or modified.
        """
        self.logger.info("Rebuilding index...")
        self.build_pipeline(force_rebuild=True)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline metrics and statistics.
        
        Returns:
            Dictionary containing metrics
        """
        metrics = self.metrics.copy()
        
        # Add component-specific statistics
        if self.data_loader:
            metrics["data_stats"] = self.data_loader.get_statistics()
        
        if self.chunker and self.metrics.get("total_chunks"):
            # Note: we don't store chunks, so can't get detailed stats
            metrics["chunk_stats"] = {
                "total_chunks": self.metrics["total_chunks"]
            }
        
        if self.embedding_generator:
            metrics["embedding_info"] = self.embedding_generator.get_model_info()
        
        if self.vector_store_manager:
            metrics["vector_store_stats"] = self.vector_store_manager.get_statistics()
        
        return metrics
    
    def _log_metrics(self) -> None:
        """Log pipeline metrics."""
        self.logger.info("Pipeline Metrics:")
        self.logger.info(f"  Total Documents: {self.metrics.get('total_documents', 0)}")
        self.logger.info(f"  Total Chunks: {self.metrics.get('total_chunks', 0)}")
        self.logger.info(f"  Embedding Model: {self.settings.embedding.model_name}")
        self.logger.info(f"  Chunk Size: {self.settings.chunking.chunk_size}")
        self.logger.info(f"  Top-K: {self.settings.retrieval.top_k}")
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status of each component
        """
        health = {
            "initialized": self.initialized,
            "data_loader": self.data_loader is not None,
            "preprocessor": self.preprocessor is not None,
            "chunker": self.chunker is not None,
            "embedding_generator": self.embedding_generator is not None,
            "vector_store": self.vector_store_manager is not None,
            "retriever": self.retriever is not None
        }
        
        # Check if vector store has data
        if self.vector_store_manager:
            health["vector_store_ready"] = (
                self.vector_store_manager.get_vector_store() is not None
            )
        
        all_healthy = all(health.values())
        health["overall"] = all_healthy
        
        self.logger.info(f"Health check: {'PASS' if all_healthy else 'FAIL'}")
        
        return health
