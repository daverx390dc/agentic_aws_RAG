"""
Main RAG Pipeline orchestrator using LangChain agents.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from agents.document_processor_agent import DocumentProcessorAgent
from agents.query_agent import QueryAgent
from agents.react_agent import ReActAgent
from config.settings import settings

class RAGPipeline:
    """Main RAG Pipeline orchestrator using LangChain agents."""
    def __init__(self):
        self.document_processor = DocumentProcessorAgent()
        self.query_agent = QueryAgent()
        self.react_agent = ReActAgent()
        logger.info("LangChain-based RAG Pipeline initialized")

    def ingest_documents(self, file_paths: List[str], source_names: Optional[List[str]] = None) -> Dict[str, Any]:
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
        return self.document_processor.process_directory(directory_path)

    def query(self, question: str, top_k: int = 5, include_sources: bool = True, agent_type: str = "rag") -> Dict[str, Any]:
        """Query the pipeline with a question using the specified agent type ('rag' or 'react')."""
        if agent_type == "react":
            try:
                response = self.react_agent.run(question)
                return {"success": True, "response": response, "agent": "react", "query": question}
            except Exception as e:
                return {"success": False, "response": str(e), "agent": "react", "query": question}
        else:
            return self.query_agent.process_query(question, top_k, include_sources)

    def batch_query(self, questions: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        return self.query_agent.batch_process_queries(questions, top_k)

    def query_with_context(self, question: str, context: str, top_k: int = 5) -> Dict[str, Any]:
        return self.query_agent.process_query_with_context(question, context, top_k)

    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        return self.query_agent.get_query_suggestions(partial_query, max_suggestions)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        return self.query_agent.analyze_query_intent(query)

    def remove_source(self, source_name: str) -> bool:
        return self.document_processor.remove_source(source_name)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        try:
            stats = self.document_processor.get_processing_stats()
            return {
                "vectorstore": stats,
                "pipeline_status": "active",
                "settings": {
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "max_tokens": settings.max_tokens,
                    "temperature": settings.temperature,
                    "vector_dimension": settings.opensearch_vector_dimension
                }
            }
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        health_status = {
            "overall_status": "healthy",
            "components": {}
        }
        try:
            stats = self.document_processor.get_processing_stats()
            health_status["components"]["vectorstore"] = {
                "status": "healthy",
                "total_documents": stats.get("total_documents", 0)
            }
        except Exception as e:
            health_status["components"]["vectorstore"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        try:
            # Test embedding generation
            test_embedding = self.query_agent.embeddings.embed_query("test")
            health_status["components"]["embeddings"] = {
                "status": "healthy",
                "embedding_dimension": len(test_embedding)
            }
        except Exception as e:
            health_status["components"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        return health_status

    def reset_pipeline(self) -> bool:
        try:
            logger.warning("Resetting entire pipeline - this will delete all documents!")
            # Implement actual reset logic if supported by vectorstore
            return True
        except Exception as e:
            logger.error(f"Error resetting pipeline: {e}")
            return False 