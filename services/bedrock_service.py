# DEPRECATED: This file is no longer used. Bedrock is now handled via LangChain's embedding and LLM wrappers.
# All logic should be migrated to use langchain_community.embeddings.BedrockEmbeddings and related classes.

raise NotImplementedError("This module is deprecated. Use LangChain's BedrockEmbeddings instead.")

"""
AWS Bedrock service for LLM and embedding operations.
"""
import json
import boto3
from typing import List, Dict, Any, Optional
from loguru import logger
from config.settings import settings


class BedrockService:
    """Service for interacting with AWS Bedrock."""
    
    def __init__(self):
        """Initialize Bedrock client."""
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        logger.info("Bedrock service initialized")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Titan embedding model."""
        embeddings = []
        
        for text in texts:
            try:
                # Prepare the request body for Titan embedding model
                request_body = {
                    "inputText": text
                }
                
                response = self.client.invoke_model(
                    modelId=settings.bedrock_embedding_model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                embedding = response_body['embedding']
                embeddings.append(embedding)
                
                logger.debug(f"Generated embedding for text of length {len(text)}")
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                raise
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def generate_response(self, prompt: str, context: str = None, max_tokens: int = None) -> str:
        """Generate response using Claude Sonnet 3."""
        if max_tokens is None:
            max_tokens = settings.max_tokens
        
        try:
            # Prepare the full prompt with context if provided
            if context:
                full_prompt = f"""Context: {context}

Question: {prompt}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, please say so."""
            else:
                full_prompt = prompt
            
            # Prepare the request body for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "temperature": settings.temperature
            }
            
            response = self.client.invoke_model(
                modelId=settings.bedrock_model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            logger.info(f"Generated response with {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_response_with_rag(self, question: str, relevant_chunks: List[str]) -> str:
        """Generate response using RAG approach."""
        # Combine relevant chunks into context
        context = "\n\n".join(relevant_chunks)
        
        # Generate response with context
        return self.generate_response(question, context) 