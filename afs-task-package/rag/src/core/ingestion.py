"""
Data ingestion module for loading documents from various sources.

This module handles loading of text documents from the file system,
with support for multiple file formats and robust error handling.
"""

import glob
import os
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from ..utils.logger import LoggerMixin


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


class DataLoader(LoggerMixin):
    """
    Data loader for ingesting documents from file system.
    
    This class provides functionality to load text documents from a specified
    directory, supporting multiple file formats with proper error handling
    and metadata extraction.
    """
    
    def __init__(
        self,
        data_dir: str = "raw_data",
        supported_formats: Optional[List[str]] = None,
        encoding: str = "utf-8"
    ):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing data files
            supported_formats: List of supported file extensions (e.g., ['.txt'])
            encoding: File encoding for reading
        """
        self.data_dir = Path(data_dir)
        self.supported_formats = supported_formats or [".txt"]
        self.encoding = encoding
        self.documents: List[Document] = []
        
        self.logger.info(f"DataLoader initialized with data_dir: {self.data_dir}")
        self.logger.debug(f"Supported formats: {self.supported_formats}")
    
    def load_documents(self) -> List[Document]:
        """
        Load all supported documents from the data directory.
        
        Returns:
            List of Document objects
            
        Raises:
            DataLoaderError: If data directory doesn't exist or no files found
        """
        if not self.data_dir.exists():
            error_msg = f"Data directory does not exist: {self.data_dir}"
            self.logger.error(error_msg)
            raise DataLoaderError(error_msg)
        
        self.logger.info(f"Loading documents from: {self.data_dir}")
        start_time = time.time()
        
        # Find all files matching supported formats
        all_files = []
        for fmt in self.supported_formats:
            pattern = os.path.join(str(self.data_dir), f"**/*{fmt}")
            files = glob.glob(pattern, recursive=True)
            all_files.extend(files)
        
        if not all_files:
            error_msg = f"No files with formats {self.supported_formats} found in {self.data_dir}"
            self.logger.warning(error_msg)
            return []
        
        self.logger.info(f"Found {len(all_files)} files to process")
        
        # Load each file
        successful_loads = 0
        failed_loads = 0
        
        for file_path in all_files:
            try:
                document = self._load_single_file(file_path)
                if document:
                    self.documents.append(document)
                    successful_loads += 1
                    self.logger.debug(f"Successfully loaded: {file_path}")
            except Exception as e:
                failed_loads += 1
                self.logger.error(f"Failed to load {file_path}: {str(e)}")
        
        load_time = time.time() - start_time
        
        self.logger.info(
            f"Document loading completed: {successful_loads} successful, "
            f"{failed_loads} failed in {load_time:.2f} seconds"
        )
        
        return self.documents
    
    def _load_single_file(self, file_path: str) -> Optional[Document]:
        """
        Load a single file and create a Document object.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            if not content.strip():
                self.logger.warning(f"Empty file: {file_path}")
                return None
            
            # Extract metadata
            path_obj = Path(file_path)
            metadata = {
                "source": str(path_obj.absolute()),
                "filename": path_obj.name,
                "category": path_obj.parent.name,
                "file_size": path_obj.stat().st_size,
                "extension": path_obj.suffix
            }
            
            document = Document(
                page_content=content,
                metadata=metadata
            )
            
            return document
            
        except UnicodeDecodeError as e:
            self.logger.error(f"Encoding error in {file_path}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading {file_path}: {str(e)}")
            return None
    
    def get_statistics(self) -> dict:
        """
        Get statistics about loaded documents.
        
        Returns:
            Dictionary containing document statistics
        """
        if not self.documents:
            return {
                "total_documents": 0,
                "total_size": 0,
                "average_size": 0
            }
        
        total_size = sum(len(doc.page_content) for doc in self.documents)
        
        stats = {
            "total_documents": len(self.documents),
            "total_size": total_size,
            "average_size": total_size / len(self.documents),
            "categories": list(set(doc.metadata.get("category", "unknown") 
                                  for doc in self.documents))
        }
        
        return stats
