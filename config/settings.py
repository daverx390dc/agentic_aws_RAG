"""
Configuration settings for the RAG pipeline.
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    # Bedrock Configuration
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v1"
    
    # OpenSearch Configuration
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_username: Optional[str] = None
    opensearch_password: Optional[str] = None
    opensearch_index_name: str = "rag_documents"
    opensearch_vector_dimension: int = 1536  # Titan embedding dimension
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4096
    temperature: float = 0.1
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings() 