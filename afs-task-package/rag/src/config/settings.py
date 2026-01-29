"""
Settings module using Pydantic for configuration management.

This module provides a centralized configuration management system using
Pydantic BaseSettings, allowing for easy validation, type checking, and
environment variable overrides.
"""

from typing import List, Optional
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings
import yaml


class DataConfig(BaseSettings):
    """Data loading configuration."""
    
    data_dir: str = Field(default="raw_data", description="Directory containing data files")
    supported_formats: List[str] = Field(default=[".txt"], description="Supported file formats")
    encoding: str = Field(default="utf-8", description="File encoding")


class PreprocessingConfig(BaseSettings):
    """Text preprocessing configuration."""
    
    remove_special_chars: bool = Field(default=True, description="Remove special characters")
    normalize_whitespace: bool = Field(default=True, description="Normalize whitespace")
    preserve_case: bool = Field(default=True, description="Preserve text case")


class ChunkingConfig(BaseSettings):
    """Document chunking configuration."""
    
    chunk_size: int = Field(default=500, description="Maximum chunk size in characters")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    separators: List[str] = Field(
        default=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        description="Separators for text splitting"
    )


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    device: str = Field(default="cpu", description="Device for model inference")
    normalize_embeddings: bool = Field(default=True, description="Normalize embeddings")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""
    
    type: str = Field(default="faiss", description="Vector store type")
    index_path: str = Field(default="faiss_index", description="Path to save/load index")
    save_on_create: bool = Field(default=True, description="Save index after creation")
    dimension: int = Field(default=384, description="Embedding dimension")


class RetrievalConfig(BaseSettings):
    """Retrieval configuration."""
    
    top_k: int = Field(default=3, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0.0, description="Minimum similarity score")
    return_scores: bool = Field(default=True, description="Return similarity scores")


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file: Optional[str] = Field(default="logs/rag_system.log", description="Log file path")
    console: bool = Field(default=True, description="Enable console logging")


class PerformanceConfig(BaseSettings):
    """Performance monitoring configuration."""
    
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    track_inference_time: bool = Field(default=True, description="Track inference time")


class SystemConfig(BaseSettings):
    """System configuration."""
    
    max_workers: int = Field(default=4, description="Maximum worker threads")
    timeout_seconds: int = Field(default=300, description="Operation timeout")
    enable_caching: bool = Field(default=True, description="Enable caching")


class Settings(BaseSettings):
    """Main settings class containing all configuration sections."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Settings":
        """
        Load settings from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Settings instance
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Return default settings if config file doesn't exist
            return cls()
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            preprocessing=PreprocessingConfig(**config_dict.get("preprocessing", {})),
            chunking=ChunkingConfig(**config_dict.get("chunking", {})),
            embedding=EmbeddingConfig(**config_dict.get("embedding", {})),
            vector_store=VectorStoreConfig(**config_dict.get("vector_store", {})),
            retrieval=RetrievalConfig(**config_dict.get("retrieval", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            performance=PerformanceConfig(**config_dict.get("performance", {})),
            system=SystemConfig(**config_dict.get("system", {})),
        )


@lru_cache()
def get_settings(config_path: str = "config.yaml") -> Settings:
    """
    Get cached settings instance.
    
    This function uses lru_cache to ensure only one Settings instance
    is created and reused throughout the application.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Cached Settings instance
    """
    return Settings.from_yaml(config_path)
