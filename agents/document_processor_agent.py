"""
Document Processor Agent for handling document ingestion and preprocessing.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import pypdf
from docx import Document
from bs4 import BeautifulSoup
from loguru import logger
from utils.text_processing import preprocess_document
from services.bedrock_service import BedrockService
from services.opensearch_service import OpenSearchService


class DocumentProcessorAgent:
    """Agent responsible for processing documents and preparing them for RAG."""
    
    def __init__(self):
        """Initialize the document processor agent."""
        self.bedrock_service = BedrockService()
        self.opensearch_service = OpenSearchService()
        logger.info("Document Processor Agent initialized")
    
    def process_file(self, file_path: str, source_name: str = None) -> Dict[str, Any]:
        """Process a single file and return processing results."""
        if source_name is None:
            source_name = Path(file_path).name
        
        try:
            # Extract text based on file type
            text = self._extract_text_from_file(file_path)
            
            # Preprocess the document
            processed_doc = preprocess_document(text, source_name)
            
            # Generate embeddings for chunks
            embeddings = self.bedrock_service.generate_embeddings(processed_doc["chunks"])
            
            # Prepare documents for indexing
            documents_to_index = []
            for i, (chunk, embedding) in enumerate(zip(processed_doc["chunks"], embeddings)):
                chunk_id = f"{source_name}_{i}_{uuid.uuid4().hex[:8]}"
                
                doc = {
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": {
                        **processed_doc["metadata"],
                        "chunk_index": i,
                        "total_chunks": len(processed_doc["chunks"])
                    },
                    "source": source_name,
                    "chunk_id": chunk_id
                }
                documents_to_index.append(doc)
            
            # Index documents
            success = self.opensearch_service.index_documents(documents_to_index)
            
            return {
                "success": success,
                "source": source_name,
                "total_chunks": len(documents_to_index),
                "file_path": file_path,
                "metadata": processed_doc["metadata"]
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "success": False,
                "source": source_name,
                "error": str(e),
                "file_path": file_path
            }
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all supported files in a directory."""
        results = []
        supported_extensions = {'.txt', '.pdf', '.docx', '.html', '.htm'}
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            files = [f for f in directory.rglob('*') if f.is_file() and f.suffix.lower() in supported_extensions]
            
            logger.info(f"Found {len(files)} files to process in {directory_path}")
            
            for file_path in files:
                result = self.process_file(str(file_path))
                results.append(result)
                
                if result["success"]:
                    logger.info(f"Successfully processed: {file_path}")
                else:
                    logger.error(f"Failed to process: {file_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            return []
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from different file types."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif extension == '.pdf':
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
            
            elif extension == '.docx':
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            elif extension in ['.html', '.htm']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    return soup.get_text()
            
            else:
                raise ValueError(f"Unsupported file type: {extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def remove_source(self, source_name: str) -> bool:
        """Remove all documents from a specific source."""
        return self.opensearch_service.delete_documents_by_source(source_name)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        return self.opensearch_service.get_index_stats() 