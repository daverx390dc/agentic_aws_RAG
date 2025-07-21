"""
Query Agent for handling user queries and generating RAG responses.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from services.bedrock_service import BedrockService
from services.opensearch_service import OpenSearchService
from utils.text_processing import clean_text


class QueryAgent:
    """Agent responsible for processing queries and generating RAG responses."""
    
    def __init__(self):
        """Initialize the query agent."""
        self.bedrock_service = BedrockService()
        self.opensearch_service = OpenSearchService()
        logger.info("Query Agent initialized")
    
    def process_query(self, query: str, top_k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """Process a user query and return a RAG response."""
        try:
            # Clean the query
            cleaned_query = clean_text(query)
            
            # Generate embedding for the query
            query_embedding = self.bedrock_service.generate_embeddings([cleaned_query])[0]
            
            # Search for similar documents
            similar_docs = self.opensearch_service.search_similar(query_embedding, k=top_k)
            
            if not similar_docs:
                return {
                    "success": False,
                    "response": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "query": query
                }
            
            # Extract relevant chunks
            relevant_chunks = [doc["content"] for doc in similar_docs]
            
            # Generate response using RAG
            response = self.bedrock_service.generate_response_with_rag(cleaned_query, relevant_chunks)
            
            # Prepare sources information
            sources = []
            if include_sources:
                for doc in similar_docs:
                    source_info = {
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "source": doc["source"],
                        "score": doc["score"],
                        "chunk_id": doc["chunk_id"]
                    }
                    sources.append(source_info)
            
            return {
                "success": True,
                "response": response,
                "sources": sources,
                "query": query,
                "num_sources": len(similar_docs),
                "avg_score": sum(doc["score"] for doc in similar_docs) / len(similar_docs) if similar_docs else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "response": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "query": query
            }
    
    def process_query_with_context(self, query: str, context: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query with additional context."""
        try:
            # Combine query with context
            enhanced_query = f"Context: {context}\n\nQuestion: {query}"
            
            # Process the enhanced query
            result = self.process_query(enhanced_query, top_k)
            
            # Add context information to the result
            result["context_used"] = context
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query with context: {e}")
            return {
                "success": False,
                "response": f"An error occurred while processing your query with context: {str(e)}",
                "sources": [],
                "query": query,
                "context_used": context
            }
    
    def batch_process_queries(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        results = []
        
        for query in queries:
            result = self.process_query(query, top_k)
            results.append(result)
            
            if result["success"]:
                logger.info(f"Successfully processed query: {query[:50]}...")
            else:
                logger.error(f"Failed to process query: {query[:50]}...")
        
        return results
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Generate query suggestions based on partial input."""
        try:
            # This is a simple implementation - in a real system, you might use
            # more sophisticated methods like query expansion or autocomplete
            suggestions = [
                f"What is {partial_query}?",
                f"Explain {partial_query}",
                f"How does {partial_query} work?",
                f"Tell me about {partial_query}",
                f"Examples of {partial_query}"
            ]
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and type of a query."""
        try:
            # Simple intent analysis - in a real system, you might use NLP techniques
            query_lower = query.lower()
            
            intent = {
                "is_question": any(word in query_lower for word in ["what", "how", "why", "when", "where", "who", "?"]),
                "is_definition": any(word in query_lower for word in ["what is", "define", "definition"]),
                "is_explanation": any(word in query_lower for word in ["explain", "how does", "tell me about"]),
                "is_comparison": any(word in query_lower for word in ["compare", "difference", "vs", "versus"]),
                "is_example": any(word in query_lower for word in ["example", "instance", "case"]),
                "query_length": len(query.split()),
                "has_technical_terms": any(word in query_lower for word in ["api", "function", "method", "class", "algorithm"])
            }
            
            return intent
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {} 