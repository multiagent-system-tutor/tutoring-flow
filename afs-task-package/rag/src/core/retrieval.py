"""
Retrieval module for similarity search and context retrieval.

This module provides functionality for querying the vector store
and retrieving relevant documents with similarity scores.
"""

import time
from typing import List, Optional, Tuple, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..utils.logger import LoggerMixin


class RetrievalError(Exception):
    """Custom exception for retrieval errors."""
    pass


class Retriever(LoggerMixin):
    """
    Retriever for similarity search and document retrieval.
    
    This class provides functionality to query the vector store
    and retrieve relevant documents based on similarity search.
    """
    
    def __init__(
        self,
        vector_store: FAISS,
        top_k: int = 3,
        score_threshold: float = 0.0,
        return_scores: bool = True
    ):
        """
        Initialize Retriever.
        
        Args:
            vector_store: FAISS vector store
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            return_scores: Whether to return similarity scores
        """
        if vector_store is None:
            raise ValueError("Vector store cannot be None")
        
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.return_scores = return_scores
        
        self.logger.info(
            f"Retriever initialized with top_k={top_k}, "
            f"score_threshold={score_threshold}"
        )
    
    def retrieve(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> Tuple[List[Document], float]:
        """
        Retrieve top-k most relevant documents.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (uses default if None)
            
        Returns:
            Tuple of (documents, retrieval_time)
            
        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            error_msg = "Empty query provided"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg)
        
        k = k or self.top_k
        
        self.logger.debug(f"Retrieving top-{k} documents for query: {query[:50]}...")
        start_time = time.time()
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            
            # Filter by score threshold if needed
            if self.score_threshold > 0.0:
                results_with_scores = self.vector_store.similarity_search_with_score(
                    query, k=k
                )
                results = [
                    doc for doc, score in results_with_scores 
                    if score >= self.score_threshold
                ]
            
            retrieval_time = time.time() - start_time
            
            self.logger.info(
                f"Retrieved {len(results)} documents in {retrieval_time*1000:.2f}ms"
            )
            
            return results, retrieval_time
            
        except Exception as e:
            error_msg = f"Error during retrieval: {str(e)}"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> Tuple[List[Tuple[Document, float]], float]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (uses default if None)
            
        Returns:
            Tuple of (list of (document, score) tuples, retrieval_time)
            
        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            error_msg = "Empty query provided"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg)
        
        k = k or self.top_k
        
        self.logger.debug(
            f"Retrieving top-{k} documents with scores for query: {query[:50]}..."
        )
        start_time = time.time()
        
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold
            if self.score_threshold > 0.0:
                results = [
                    (doc, score) for doc, score in results 
                    if score >= self.score_threshold
                ]
            
            retrieval_time = time.time() - start_time
            
            # Log scores
            score_info = ", ".join([f"{score:.4f}" for _, score in results[:3]])
            self.logger.info(
                f"Retrieved {len(results)} documents with scores [{score_info}...] "
                f"in {retrieval_time*1000:.2f}ms"
            )
            
            return results, retrieval_time
            
        except Exception as e:
            error_msg = f"Error during retrieval with scores: {str(e)}"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def format_context(
        self, 
        results: List[Tuple[Document, float]]
    ) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            results: List of (document, score) tuples
            
        Returns:
            Formatted context string
        """
        if not results:
            self.logger.warning("No results to format")
            return ""
        
        context_parts = []
        
        for i, (doc, score) in enumerate(results):
            context_parts.append(f"[Source {i+1} - Score: {score:.4f}]")
            context_parts.append(doc.page_content)
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def process_query(
        self,
        student_input: str,
        summary: str = "",
        student_profile: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a complete query with student input and profile.
        
        Args:
            student_input: Student's input/answer
            summary: Summary of student's work
            student_profile: Student profile information
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing context and metadata
            
        Raises:
            RetrievalError: If processing fails
        """
        self.logger.info("Processing query with student context")
        
        try:
            # Build enhanced query
            query_parts = []
            
            if summary:
                query_parts.append(f"Summary: {summary}")
            
            query_parts.append(f"Student Input: {student_input}")
            
            if student_profile:
                level = student_profile.get('level', 'N/A')
                query_parts.append(f"Level: {level}")
            
            enhanced_query = " | ".join(query_parts)
            
            # Retrieve documents
            results, retrieval_time = self.retrieve_with_scores(enhanced_query, k=k)
            
            # Format context
            context_text = self.format_context(results)
            
            # Build output
            output = {
                "context": context_text,
                "student_input": student_input,
                "summary": summary,
                "student_profile": student_profile,
                "retrieval_time": retrieval_time,
                "num_sources": len(results),
                "sources": [
                    {
                        "filename": doc.metadata.get('filename', 'N/A'),
                        "category": doc.metadata.get('category', 'N/A'),
                        "score": float(score),
                        "preview": doc.page_content[:150]
                    }
                    for doc, score in results
                ]
            }
            
            self.logger.info(
                f"Query processed successfully: {len(results)} sources retrieved "
                f"in {retrieval_time*1000:.2f}ms"
            )
            
            return output
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg)
            raise RetrievalError(error_msg) from e
    
    def get_statistics(self, results: List[Tuple[Document, float]]) -> dict:
        """
        Get statistics about retrieval results.
        
        Args:
            results: List of (document, score) tuples
            
        Returns:
            Dictionary containing retrieval statistics
        """
        if not results:
            return {
                "num_results": 0,
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0
            }
        
        scores = [score for _, score in results]
        
        stats = {
            "num_results": len(results),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_threshold": self.score_threshold
        }
        
        return stats
