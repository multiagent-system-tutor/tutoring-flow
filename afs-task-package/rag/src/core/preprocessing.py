"""
Text preprocessing module for cleaning and normalizing text data.

This module provides functionality for text preprocessing including
cleaning, normalization, and noise removal from documents.
"""

import re
import time
from typing import List

from langchain_core.documents import Document

from ..utils.logger import LoggerMixin


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


class TextPreprocessor(LoggerMixin):
    """
    Text preprocessor for cleaning and normalizing document content.
    
    This class provides various text preprocessing operations to clean
    and normalize text before chunking and embedding.
    """
    
    def __init__(
        self,
        remove_special_chars: bool = True,
        normalize_whitespace: bool = True,
        preserve_case: bool = True
    ):
        """
        Initialize TextPreprocessor.
        
        Args:
            remove_special_chars: Remove special characters
            normalize_whitespace: Normalize whitespace
            preserve_case: Preserve text case (useful for educational content)
        """
        self.remove_special_chars = remove_special_chars
        self.normalize_whitespace = normalize_whitespace
        self.preserve_case = preserve_case
        
        self.logger.info("TextPreprocessor initialized")
        self.logger.debug(
            f"Config: remove_special_chars={remove_special_chars}, "
            f"normalize_whitespace={normalize_whitespace}, "
            f"preserve_case={preserve_case}"
        )
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of preprocessed Document objects
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        if not documents:
            self.logger.warning("No documents to preprocess")
            return []
        
        self.logger.info(f"Starting preprocessing of {len(documents)} documents")
        start_time = time.time()
        
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                processed_text = self._preprocess_text(doc.page_content)
                
                # Create new document with processed text
                processed_doc = Document(
                    page_content=processed_text,
                    metadata=doc.metadata.copy()
                )
                processed_docs.append(processed_doc)
                
            except Exception as e:
                self.logger.error(
                    f"Error preprocessing document {i} "
                    f"({doc.metadata.get('filename', 'unknown')}): {str(e)}"
                )
                # Keep original document if preprocessing fails
                processed_docs.append(doc)
        
        preprocess_time = time.time() - start_time
        
        self.logger.info(
            f"Preprocessing completed: {len(processed_docs)} documents "
            f"processed in {preprocess_time:.2f} seconds"
        )
        
        return processed_docs
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Apply cleaning
        if self.remove_special_chars:
            text = self._clean_text(text)
        
        # Apply normalization
        if self.normalize_whitespace:
            text = self._normalize_text(text)
        
        # Apply case transformation if needed
        if not self.preserve_case:
            text = text.lower()
        
        return text.strip()
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing noise and special characters.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove multiple whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep alphanumeric and common punctuation)
        text = re.sub(r'[^\w\s\.,!?;:()\-\'\"]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n\n\n', '\n\n')
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text formatting.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def get_statistics(self, original_docs: List[Document], 
                      processed_docs: List[Document]) -> dict:
        """
        Get statistics about preprocessing impact.
        
        Args:
            original_docs: Original documents
            processed_docs: Processed documents
            
        Returns:
            Dictionary containing preprocessing statistics
        """
        if not original_docs or not processed_docs:
            return {}
        
        original_size = sum(len(doc.page_content) for doc in original_docs)
        processed_size = sum(len(doc.page_content) for doc in processed_docs)
        
        reduction = ((original_size - processed_size) / original_size * 100 
                    if original_size > 0 else 0)
        
        stats = {
            "original_size": original_size,
            "processed_size": processed_size,
            "size_reduction_percent": reduction,
            "documents_processed": len(processed_docs)
        }
        
        return stats
