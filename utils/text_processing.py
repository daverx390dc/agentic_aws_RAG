"""
Text processing utilities for the RAG pipeline.
"""
import re
from typing import List, Dict, Any
from loguru import logger
from config.settings import settings


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """Split text into overlapping chunks."""
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is not the first chunk, include some overlap
        if start > 0:
            start = start - chunk_overlap
        
        # Extract the chunk
        chunk = text[start:end]
        
        # Clean the chunk
        chunk = clean_text(chunk)
        
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk
        start = end
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def extract_metadata(text: str, source: str = None) -> Dict[str, Any]:
    """Extract metadata from text."""
    metadata = {
        "source": source,
        "length": len(text),
        "word_count": len(text.split()),
        "chunk_count": len(split_text_into_chunks(text))
    }
    return metadata


def preprocess_document(text: str, source: str = None) -> Dict[str, Any]:
    """Preprocess a document for the RAG pipeline."""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Split into chunks
    chunks = split_text_into_chunks(cleaned_text)
    
    # Extract metadata
    metadata = extract_metadata(cleaned_text, source)
    
    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "chunks": chunks,
        "metadata": metadata
    } 