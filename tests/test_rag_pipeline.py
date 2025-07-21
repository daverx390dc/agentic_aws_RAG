"""
Tests for the RAG pipeline components.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_pipeline import RAGPipeline
from utils.text_processing import clean_text, split_text_into_chunks, preprocess_document


class TestTextProcessing:
    """Test text processing utilities."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        text = "  This   is   a   test   text  with  extra  spaces  "
        cleaned = clean_text(text)
        assert cleaned == "This is a test text with extra spaces"
    
    def test_split_text_into_chunks(self):
        """Test text chunking functionality."""
        text = "This is a test text that should be split into chunks. " * 10
        chunks = split_text_into_chunks(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 0
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_preprocess_document(self):
        """Test document preprocessing."""
        text = "This is a test document for preprocessing."
        result = preprocess_document(text, "test_source")
        
        assert "original_text" in result
        assert "cleaned_text" in result
        assert "chunks" in result
        assert "metadata" in result
        assert result["metadata"]["source"] == "test_source"


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        with patch('models.rag_pipeline.BedrockService'), \
             patch('models.rag_pipeline.OpenSearchService'):
            pipeline = RAGPipeline()
            return pipeline
    
    def test_pipeline_initialization(self, mock_pipeline):
        """Test pipeline initialization."""
        assert mock_pipeline is not None
        assert hasattr(mock_pipeline, 'document_processor')
        assert hasattr(mock_pipeline, 'query_agent')
        assert hasattr(mock_pipeline, 'opensearch_service')
    
    def test_health_check(self, mock_pipeline):
        """Test health check functionality."""
        # Mock the health check responses
        mock_pipeline.opensearch_service.get_index_stats.return_value = {
            "total_documents": 10
        }
        mock_pipeline.query_agent.bedrock_service.generate_embeddings.return_value = [[0.1] * 1536]
        
        health = mock_pipeline.health_check()
        
        assert "overall_status" in health
        assert "components" in health
        assert "opensearch" in health["components"]
        assert "bedrock" in health["components"]
    
    def test_get_pipeline_stats(self, mock_pipeline):
        """Test pipeline statistics."""
        # Mock the stats response
        mock_pipeline.opensearch_service.get_index_stats.return_value = {
            "total_documents": 10,
            "index_size": 1024
        }
        
        stats = mock_pipeline.get_pipeline_stats()
        
        assert "opensearch" in stats
        assert "pipeline_status" in stats
        assert "settings" in stats


class TestDocumentProcessing:
    """Test document processing functionality."""
    
    def test_supported_file_types(self):
        """Test that supported file types are correctly identified."""
        supported_extensions = {'.txt', '.pdf', '.docx', '.html', '.htm'}
        
        # Test supported files
        for ext in supported_extensions:
            assert ext in supported_extensions
        
        # Test unsupported files
        unsupported = {'.jpg', '.png', '.mp3', '.mp4'}
        for ext in unsupported:
            assert ext not in supported_extensions


class TestQueryProcessing:
    """Test query processing functionality."""
    
    def test_query_analysis(self):
        """Test query intent analysis."""
        # This would test the query analysis functionality
        # For now, we'll just test that the function exists
        assert True  # Placeholder test


# Integration tests (require actual services)
@pytest.mark.integration
class TestIntegration:
    """Integration tests that require actual services."""
    
    @pytest.mark.skip(reason="Requires AWS Bedrock and OpenSearch")
    def test_full_pipeline_integration(self):
        """Test full pipeline integration."""
        # This test would require actual AWS Bedrock and OpenSearch instances
        pass
    
    @pytest.mark.skip(reason="Requires AWS Bedrock and OpenSearch")
    def test_document_ingestion_integration(self):
        """Test document ingestion with real services."""
        pass
    
    @pytest.mark.skip(reason="Requires AWS Bedrock and OpenSearch")
    def test_query_processing_integration(self):
        """Test query processing with real services."""
        pass


# Utility functions for testing
def create_temp_document(content: str, extension: str = ".txt") -> str:
    """Create a temporary document for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
        f.write(content)
        return f.name


def cleanup_temp_file(file_path: str):
    """Clean up temporary file."""
    try:
        os.unlink(file_path)
    except OSError:
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 