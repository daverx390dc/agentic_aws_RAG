# DEPRECATED: This file is no longer used. OpenSearch is now handled via LangChain's vectorstore integrations.
# All logic should be migrated to use langchain_community.vectorstores.OpenSearchVectorSearch.

raise NotImplementedError("This module is deprecated. Use LangChain's OpenSearchVectorSearch instead.")

"""
OpenSearch service for vector storage and retrieval.
"""
from opensearchpy import OpenSearch
from typing import List, Dict, Any, Optional
import json
from loguru import logger
from config.settings import settings


class OpenSearchService:
    """Service for interacting with OpenSearch."""
    
    def __init__(self):
        """Initialize OpenSearch client."""
        # Prepare connection parameters
        auth = None
        if settings.opensearch_username and settings.opensearch_password:
            auth = (settings.opensearch_username, settings.opensearch_password)
        
        self.client = OpenSearch(
            hosts=[{'host': settings.opensearch_host, 'port': settings.opensearch_port}],
            http_auth=auth,
            use_ssl=False,  # Set to True if using HTTPS
            verify_certs=False,  # Set to True if using HTTPS
            ssl_show_warn=False
        )
        
        # Ensure index exists
        self._create_index_if_not_exists()
        logger.info("OpenSearch service initialized")
    
    def _create_index_if_not_exists(self):
        """Create the index if it doesn't exist."""
        if not self.client.indices.exists(index=settings.opensearch_index_name):
            # Define index mapping for vector search
            index_mapping = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text"
                        },
                        "content_vector": {
                            "type": "knn_vector",
                            "dimension": settings.opensearch_vector_dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 16
                                }
                            }
                        },
                        "metadata": {
                            "type": "object"
                        },
                        "source": {
                            "type": "keyword"
                        },
                        "chunk_id": {
                            "type": "keyword"
                        }
                    }
                },
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                }
            }
            
            self.client.indices.create(
                index=settings.opensearch_index_name,
                body=index_mapping
            )
            logger.info(f"Created index: {settings.opensearch_index_name}")
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents with their embeddings."""
        try:
            for doc in documents:
                # Prepare document for indexing
                index_doc = {
                    "content": doc["content"],
                    "content_vector": doc["embedding"],
                    "metadata": doc.get("metadata", {}),
                    "source": doc.get("source", "unknown"),
                    "chunk_id": doc.get("chunk_id", "")
                }
                
                # Index the document
                self.client.index(
                    index=settings.opensearch_index_name,
                    body=index_doc,
                    id=doc.get("chunk_id", None)
                )
            
            # Refresh index to make documents searchable
            self.client.indices.refresh(index=settings.opensearch_index_name)
            logger.info(f"Indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        try:
            # Prepare search query
            search_query = {
                "size": k,
                "query": {
                    "knn": {
                        "content_vector": {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                },
                "_source": ["content", "metadata", "source", "chunk_id"]
            }
            
            # Execute search
            response = self.client.search(
                index=settings.opensearch_index_name,
                body=search_query
            )
            
            # Extract results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"]["metadata"],
                    "source": hit["_source"]["source"],
                    "chunk_id": hit["_source"]["chunk_id"],
                    "score": hit["_score"]
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def delete_documents_by_source(self, source: str) -> bool:
        """Delete all documents from a specific source."""
        try:
            # Delete by query
            delete_query = {
                "query": {
                    "term": {
                        "source": source
                    }
                }
            }
            
            response = self.client.delete_by_query(
                index=settings.opensearch_index_name,
                body=delete_query
            )
            
            logger.info(f"Deleted {response['deleted']} documents from source: {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self.client.indices.stats(index=settings.opensearch_index_name)
            return {
                "total_documents": stats["indices"][settings.opensearch_index_name]["total"]["docs"]["count"],
                "index_size": stats["indices"][settings.opensearch_index_name]["total"]["store"]["size_in_bytes"]
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {} 