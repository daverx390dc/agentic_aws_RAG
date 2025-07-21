"""
Main entry point for the RAG Pipeline application.
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.rag_pipeline import RAGPipeline
from utils.logger import setup_logger
from loguru import logger


def main():
    """Main function to run the RAG pipeline."""
    # Setup logging
    setup_logger()
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logger.info("Starting RAG Pipeline")
    
    try:
        # Initialize the pipeline
        pipeline = RAGPipeline()
        
        # Perform health check
        health_status = pipeline.health_check()
        logger.info(f"Pipeline health status: {health_status['overall_status']}")
        
        if health_status['overall_status'] == 'healthy':
            logger.info("RAG Pipeline is ready to use!")
            logger.info("Use the CLI or API to interact with the pipeline.")
            logger.info("CLI: python scripts/cli.py --help")
            logger.info("API: python services/api_service.py")
        else:
            logger.error("Pipeline health check failed!")
            for component, status in health_status['components'].items():
                if status.get('status') == 'unhealthy':
                    logger.error(f"{component}: {status.get('error', 'Unknown error')}")
            
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error starting RAG Pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 