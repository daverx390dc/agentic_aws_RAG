"""
FastAPI service for the RAG pipeline.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import tempfile
import os
from pathlib import Path
from loguru import logger
from models.rag_pipeline import RAGPipeline
from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation Pipeline with AWS Bedrock and OpenSearch",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    include_sources: bool = True

class QueryResponse(BaseModel):
    success: bool
    response: str
    sources: List[Dict[str, Any]]
    query: str
    num_sources: int
    avg_score: float

class BatchQueryRequest(BaseModel):
    questions: List[str]
    top_k: int = 5

class IngestRequest(BaseModel):
    file_paths: List[str]
    source_names: Optional[List[str]] = None

class HealthResponse(BaseModel):
    overall_status: str
    components: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return rag_pipeline.health_check()

@app.get("/stats")
async def get_stats():
    """Get pipeline statistics."""
    return rag_pipeline.get_pipeline_stats()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG pipeline."""
    try:
        result = rag_pipeline.query(
            request.question,
            request.top_k,
            request.include_sources
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/batch")
async def batch_query(request: BatchQueryRequest):
    """Process multiple queries in batch."""
    try:
        results = rag_pipeline.batch_query(request.questions, request.top_k)
        return {
            "total_queries": len(request.questions),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in batch query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/with-context")
async def query_with_context(
    question: str = Form(...),
    context: str = Form(...),
    top_k: int = Form(5)
):
    """Query with additional context."""
    try:
        result = rag_pipeline.query_with_context(question, context, top_k)
        return result
    except Exception as e:
        logger.error(f"Error in query with context endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggestions")
async def get_suggestions(partial_query: str, max_suggestions: int = 5):
    """Get query suggestions."""
    try:
        suggestions = rag_pipeline.get_query_suggestions(partial_query, max_suggestions)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error in suggestions endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-query")
async def analyze_query(query: str):
    """Analyze query intent and characteristics."""
    try:
        analysis = rag_pipeline.analyze_query(query)
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Error in analyze query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/files")
async def ingest_files(request: IngestRequest):
    """Ingest multiple files."""
    try:
        results = rag_pipeline.ingest_documents(request.file_paths, request.source_names)
        return results
    except Exception as e:
        logger.error(f"Error in ingest files endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None)
):
    """Upload and ingest a single file."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the file
        result = rag_pipeline.ingest_documents([temp_file_path], [source_name or file.filename])
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error in upload and ingest endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/directory")
async def ingest_directory(directory_path: str):
    """Ingest all supported files from a directory."""
    try:
        results = rag_pipeline.ingest_directory(directory_path)
        return {
            "directory_path": directory_path,
            "total_files": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in ingest directory endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sources/{source_name}")
async def remove_source(source_name: str):
    """Remove all documents from a specific source."""
    try:
        success = rag_pipeline.remove_source(source_name)
        if success:
            return {"message": f"Successfully removed source: {source_name}"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to remove source: {source_name}")
    except Exception as e:
        logger.error(f"Error in remove source endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_pipeline():
    """Reset the entire pipeline."""
    try:
        success = rag_pipeline.reset_pipeline()
        if success:
            return {"message": "Pipeline reset successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset pipeline")
    except Exception as e:
        logger.error(f"Error in reset pipeline endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.api_service:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    ) 