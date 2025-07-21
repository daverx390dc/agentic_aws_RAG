"""
Example usage of the RAG Pipeline.

This script demonstrates how to use the RAG pipeline programmatically.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rag_pipeline import RAGPipeline
from loguru import logger


def main():
    """Demonstrate RAG pipeline usage."""
    logger.info("Starting RAG Pipeline example")
    
    try:
        # Initialize the pipeline
        pipeline = RAGPipeline()
        
        # Check health
        health = pipeline.health_check()
        logger.info(f"Pipeline health: {health['overall_status']}")
        
        if health['overall_status'] != 'healthy':
            logger.error("Pipeline is not healthy. Please check your configuration.")
            return
        
        # Example 1: Ingest a document
        logger.info("Example 1: Ingesting a document")
        example_doc = "data/example_document.txt"
        
        if Path(example_doc).exists():
            results = pipeline.ingest_documents([example_doc])
            logger.info(f"Ingestion results: {results['successful']} successful, {results['failed']} failed")
        else:
            logger.warning(f"Example document not found: {example_doc}")
        
        # Example 2: Query the pipeline
        logger.info("Example 2: Querying the pipeline")
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the different types of machine learning?",
            "What is deep learning?"
        ]
        
        for question in questions:
            logger.info(f"\nQuestion: {question}")
            response = pipeline.query(question, top_k=3)
            
            if response['success']:
                logger.info(f"Response: {response['response'][:200]}...")
                logger.info(f"Sources: {response['num_sources']}")
            else:
                logger.error(f"Failed to get response: {response['response']}")
        
        # Example 3: Batch query
        logger.info("\nExample 3: Batch query")
        batch_results = pipeline.batch_query(questions[:2])
        logger.info(f"Batch processed {len(batch_results)} questions")
        
        # Example 4: Get query suggestions
        logger.info("\nExample 4: Query suggestions")
        suggestions = pipeline.get_query_suggestions("machine learning", max_suggestions=3)
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"{i}. {suggestion}")
        
        # Example 5: Analyze query
        logger.info("\nExample 5: Query analysis")
        analysis = pipeline.analyze_query("What is the difference between supervised and unsupervised learning?")
        for key, value in analysis.items():
            logger.info(f"{key}: {value}")
        
        # Example 6: Get pipeline statistics
        logger.info("\nExample 6: Pipeline statistics")
        stats = pipeline.get_pipeline_stats()
        logger.info(f"Total documents: {stats.get('vectorstore', {}).get('total_documents', 0)}")
        
        logger.info("\nExample completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 