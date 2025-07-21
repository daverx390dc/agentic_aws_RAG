"""
Main RAG Pipeline orchestrator.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from agents.document_processor_agent import DocumentProcessorAgent
from agents.query_agent import QueryAgent
from services.opensearch_service import OpenSearchService
from config.settings import settings


class RAGPipeline:
    """Main RAG Pipeline orchestrator."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.document_processor = DocumentProcessorAgent()
        self.query_agent = QueryAgent()
        self.opensearch_service = OpenSearchService()
        logger.info("RAG Pipeline initialized")
    
    def ingest_documents(self, file_paths: List[str], source_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Ingest multiple documents into the RAG pipeline."""
        results = {
            "total_files": len(file_paths),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        for i, file_path in enumerate(file_paths):
            source_name = source_names[i] if source_names and i < len(source_names) else None
            result = self.document_processor.process_file(file_path, source_name)
            
            results["results"].append(result)
            
            if result["success"]:
                results["successful"] += 1
                logger.info(f"Successfully ingested: {file_path}")
            else:
                results["failed"] += 1
                logger.error(f"Failed to ingest: {file_path}")
        
        logger.info(f"Ingestion completed: {results['successful']} successful, {results['failed']} failed")
        return results
    
    def ingest_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Ingest all supported documents from a directory."""
        return self.document_processor.process_directory(directory_path)
    
    def query(self, question: str, top_k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG pipeline with a question."""
        return self.query_agent.process_query(question, top_k, include_sources)
    
    def batch_query(self, questions: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        return self.query_agent.batch_process_queries(questions, top_k)
    
    def query_with_context(self, question: str, context: str, top_k: int = 5) -> Dict[str, Any]:
        """Query with additional context."""
        return self.query_agent.process_query_with_context(question, context, top_k)
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Get query suggestions."""
        return self.query_agent.get_query_suggestions(partial_query, max_suggestions)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and characteristics."""
        return self.query_agent.analyze_query_intent(query)
    
    def remove_source(self, source_name: str) -> bool:
        """Remove all documents from a specific source."""
        return self.document_processor.remove_source(source_name)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        try:
            opensearch_stats = self.opensearch_service.get_index_stats()
            
            stats = {
                "opensearch": opensearch_stats,
                "pipeline_status": "active",
                "settings": {
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "max_tokens": settings.max_tokens,
                    "temperature": settings.temperature,
                    "vector_dimension": settings.opensearch_vector_dimension
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all pipeline components."""
        health_status = {
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Check OpenSearch connection
            stats = self.opensearch_service.get_index_stats()
            health_status["components"]["opensearch"] = {
                "status": "healthy",
                "total_documents": stats.get("total_documents", 0)
            }
        except Exception as e:
            health_status["components"]["opensearch"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        
        try:
            # Check Bedrock service (simple test)
            test_embedding = self.query_agent.bedrock_service.generate_embeddings(["test"])[0]
            health_status["components"]["bedrock"] = {
                "status": "healthy",
                "embedding_dimension": len(test_embedding)
            }
        except Exception as e:
            health_status["components"]["bedrock"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        
        return health_status
    
    def reset_pipeline(self) -> bool:
        """Reset the entire pipeline (delete all documents)."""
        try:
            # This would delete all documents from the index
            # In a production system, you might want to be more careful about this
            logger.warning("Resetting entire pipeline - this will delete all documents!")
            
            # For now, we'll just return success - implement actual reset logic as needed
            return True
            
        except Exception as e:
            logger.error(f"Error resetting pipeline: {e}")
            return False 