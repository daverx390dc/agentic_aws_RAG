"""
Query Agent using LangChain for handling user queries and generating RAG responses.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from config.settings import settings

class QueryAgent:
    """Agent responsible for processing queries and generating RAG responses using LangChain."""
    def __init__(self):
        self.embeddings = BedrockEmbeddings()
        self.vectorstore = OpenSearchVectorSearch(
            opensearch_url=settings.opensearch_url,
            index_name=settings.opensearch_index,
            embedding_function=self.embeddings,
            http_auth=(settings.opensearch_user, settings.opensearch_password),
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.embeddings.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": settings.top_k}),
        )
        logger.info("LangChain-based Query Agent initialized")

    def process_query(self, query: str, top_k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """Process a user query and return a RAG response using LangChain."""
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            result = self.qa_chain.run(query, retriever=retriever)
            sources = []
            if include_sources:
                docs = retriever.get_relevant_documents(query)
                for doc in docs:
                    sources.append({
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "metadata": doc.metadata
                    })
            return {
                "success": True,
                "response": result,
                "sources": sources,
                "query": query,
                "num_sources": len(sources)
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
        """Process a query with additional context using LangChain."""
        try:
            enhanced_query = f"Context: {context}\n\nQuestion: {query}"
            return self.process_query(enhanced_query, top_k)
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
        """Process multiple queries in batch using LangChain."""
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
        suggestions = [
            f"What is {partial_query}?",
            f"Explain {partial_query}",
            f"How does {partial_query} work?",
            f"Tell me about {partial_query}",
            f"Examples of {partial_query}"
        ]
        return suggestions[:max_suggestions]

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze the intent and type of a query (simple heuristic)."""
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