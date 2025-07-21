"""
RAG Pipeline - A comprehensive Retrieval-Augmented Generation system.

This package provides a complete RAG pipeline implementation using:
- AWS Bedrock (Claude Sonnet 3 + Titan embeddings)
- OpenSearch for vector storage
- AI agents for document processing and query handling
- REST API and CLI interfaces
"""

__version__ = "1.0.0"
__author__ = "RAG Pipeline Team"
__description__ = "A comprehensive RAG pipeline with AWS Bedrock and OpenSearch"

from .models.rag_pipeline import RAGPipeline

__all__ = ["RAGPipeline"] 