"""
Document chunking module for splitting documents into manageable pieces.

This module provides functionality for splitting documents into chunks
with configurable size and overlap for optimal retrieval performance.
"""

import time
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..utils.logger import LoggerMixin


class ChunkingError(Exception):
    """Custom exception for chunking errors."""
    pass


class DocumentChunker(LoggerMixin):
    """
    Document chunker for splitting documents into chunks.
    
    This class uses RecursiveCharacterTextSplitter to split documents
    into chunks with overlap, maintaining context continuity.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        Initialize DocumentChunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self.separators
        )
        
        self.logger.info(
            f"DocumentChunker initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
        self.logger.debug(f"Separators: {self.separators}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
            
        Raises:
            ChunkingError: If chunking fails
        """
        if not documents:
            self.logger.warning("No documents to chunk")
            return []
        
        self.logger.info(f"Starting chunking of {len(documents)} documents")
        start_time = time.time()
        
        try:
            # Split documents using LangChain's text splitter
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            chunk_time = time.time() - start_time
            
            avg_chunks = len(chunks) / len(documents) if documents else 0
            
            self.logger.info(
                f"Chunking completed: {len(documents)} documents split into "
                f"{len(chunks)} chunks ({avg_chunks:.1f} avg per doc) "
                f"in {chunk_time:.2f} seconds"
            )
            
            return chunks
            
        except Exception as e:
            error_msg = f"Error during chunking: {str(e)}"
            self.logger.error(error_msg)
            raise ChunkingError(error_msg) from e
    
    def get_statistics(self, chunks: List[Document]) -> dict:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunk Document objects
            
        Returns:
            Dictionary containing chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "average_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_size": sum(chunk_sizes)
        }
        
        return stats
    
    def validate_chunks(self, chunks: List[Document]) -> bool:
        """
        Validate that chunks meet size requirements.
        
        Args:
            chunks: List of chunk Document objects
            
        Returns:
            True if all chunks are valid
        """
        for i, chunk in enumerate(chunks):
            chunk_size = len(chunk.page_content)
            
            # Check if chunk exceeds maximum size
            if chunk_size > self.chunk_size * 1.5:  # Allow 50% tolerance
                self.logger.warning(
                    f"Chunk {i} exceeds size limit: {chunk_size} > {self.chunk_size}"
                )
                return False
            
            # Check if chunk is too small (except last chunk)
            if chunk_size < 10 and i < len(chunks) - 1:
                self.logger.warning(f"Chunk {i} is too small: {chunk_size} characters")
                return False
        
        self.logger.debug(f"All {len(chunks)} chunks validated successfully")
        return True
