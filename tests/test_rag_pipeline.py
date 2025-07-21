"""
Tests for the RAG pipeline components (LangChain version).
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rag_pipeline import RAGPipeline

class TestRAGPipeline:
    """Test RAG pipeline functionality (LangChain version)."""
    @pytest.fixture
    def mock_pipeline(self):
        with patch('agents.document_processor_agent.BedrockEmbeddings'), \
             patch('agents.document_processor_agent.OpenSearchVectorSearch'), \
             patch('agents.query_agent.BedrockEmbeddings'), \
             patch('agents.query_agent.OpenSearchVectorSearch'), \
             patch('agents.query_agent.RetrievalQA'):
            pipeline = RAGPipeline()
            # Patch vectorstore and embeddings for stats/health
            pipeline.document_processor.vectorstore = MagicMock()
            pipeline.document_processor.vectorstore._client.count.return_value = {'count': 42}
            pipeline.query_agent.embeddings.embed_query = MagicMock(return_value=[0.1]*1536)
            return pipeline

    def test_pipeline_initialization(self, mock_pipeline):
        assert mock_pipeline is not None
        assert hasattr(mock_pipeline, 'document_processor')
        assert hasattr(mock_pipeline, 'query_agent')

    def test_health_check(self, mock_pipeline):
        health = mock_pipeline.health_check()
        assert "overall_status" in health
        assert "components" in health
        assert "vectorstore" in health["components"]
        assert "embeddings" in health["components"]
        assert health["components"]["vectorstore"]["status"] == "healthy"
        assert health["components"]["embeddings"]["status"] == "healthy"

    def test_get_pipeline_stats(self, mock_pipeline):
        stats = mock_pipeline.get_pipeline_stats()
        assert "vectorstore" in stats
        assert "pipeline_status" in stats
        assert "settings" in stats
        assert stats["vectorstore"]["total_documents"] == 42

    def test_remove_source(self, mock_pipeline):
        # Patch remove_source to always return True
        mock_pipeline.document_processor.remove_source = MagicMock(return_value=True)
        assert mock_pipeline.remove_source("test_source") is True

    def test_query_suggestions(self, mock_pipeline):
        suggestions = mock_pipeline.get_query_suggestions("test", max_suggestions=2)
        assert isinstance(suggestions, list)
        assert len(suggestions) == 2

    def test_analyze_query(self, mock_pipeline):
        analysis = mock_pipeline.analyze_query("What is AI?")
        assert isinstance(analysis, dict)
        assert analysis["is_question"] is True

# Utility functions for testing
def create_temp_document(content: str, extension: str = ".txt") -> str:
    with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
        f.write(content)
        return f.name

def cleanup_temp_file(file_path: str):
    try:
        os.unlink(file_path)
    except OSError:
        pass

if __name__ == "__main__":
    pytest.main([__file__]) 