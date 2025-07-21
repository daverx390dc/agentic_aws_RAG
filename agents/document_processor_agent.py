"""
Document Processor Agent using LangChain for document ingestion and preprocessing.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from config.settings import settings

class DocumentProcessorAgent:
    """Agent responsible for processing documents and preparing them for RAG using LangChain."""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.embeddings = BedrockEmbeddings()
        self.vectorstore = OpenSearchVectorSearch(
            opensearch_url=settings.opensearch_url,
            index_name=settings.opensearch_index,
            embedding_function=self.embeddings,
            http_auth=(settings.opensearch_user, settings.opensearch_password),
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs
        )
        logger.info("LangChain-based Document Processor Agent initialized")

    def process_file(self, file_path: str, source_name: str = None) -> Dict[str, Any]:
        """Process a single file and return processing results using LangChain."""
        if source_name is None:
            source_name = Path(file_path).name
        try:
            docs = self._load_document(file_path, source_name)
            if not docs:
                raise ValueError("No content loaded from file.")
            splits = self.text_splitter.split_documents(docs)
            self.vectorstore.add_documents(splits)
            return {
                "success": True,
                "source": source_name,
                "total_chunks": len(splits),
                "file_path": file_path,
                "metadata": splits[0].metadata if splits else {}
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
        """Process all supported files in a directory using LangChain loaders."""
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

    def _load_document(self, file_path: str, source_name: str):
        """Load document using LangChain loaders."""
        ext = Path(file_path).suffix.lower()
        if ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext in ['.html', '.htm']:
            loader = UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = source_name
        return docs

    def remove_source(self, source_name: str) -> bool:
        """Remove all documents from a specific source using vectorstore delete."""
        # This assumes the vectorstore supports deletion by metadata
        try:
            self.vectorstore.delete(filter={"source": source_name})
            return True
        except Exception as e:
            logger.error(f"Error removing source {source_name}: {e}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents from vectorstore."""
        try:
            stats = self.vectorstore._client.count(index=settings.opensearch_index)
            return {"total_documents": stats['count']}
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {"error": str(e)} 