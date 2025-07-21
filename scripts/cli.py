"""
Command-line interface for the RAG pipeline.
"""
import click
import json
from pathlib import Path
from typing import List
from loguru import logger
from models.rag_pipeline import RAGPipeline


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """RAG Pipeline CLI - A powerful RAG system with AWS Bedrock and OpenSearch."""
    if verbose:
        logger.info("Verbose logging enabled")


@cli.command()
@click.argument('question')
@click.option('--top-k', default=5, help='Number of top results to retrieve')
@click.option('--include-sources', is_flag=True, default=True, help='Include source information')
@click.option('--output', '-o', help='Output file for results')
def query(question, top_k, include_sources, output):
    """Query the RAG pipeline with a question."""
    try:
        pipeline = RAGPipeline()
        result = pipeline.query(question, top_k, include_sources)
        
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(f"Question: {result['query']}")
            click.echo(f"Response: {result['response']}")
            if result['sources']:
                click.echo(f"\nSources ({result['num_sources']}):")
                for i, source in enumerate(result['sources'], 1):
                    click.echo(f"{i}. {source['source']} (score: {source['score']:.3f})")
                    click.echo(f"   {source['content'][:100]}...")
                    click.echo()
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument('file_paths', nargs=-1)
@click.option('--source-names', '-s', multiple=True, help='Source names for the files')
@click.option('--output', '-o', help='Output file for results')
def ingest(file_paths, source_names, output):
    """Ingest documents into the RAG pipeline."""
    try:
        pipeline = RAGPipeline()
        
        # Convert source_names to list if provided
        source_names_list = list(source_names) if source_names else None
        
        result = pipeline.ingest_documents(list(file_paths), source_names_list)
        
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(f"Ingestion completed:")
            click.echo(f"  Total files: {result['total_files']}")
            click.echo(f"  Successful: {result['successful']}")
            click.echo(f"  Failed: {result['failed']}")
            
            if result['failed'] > 0:
                click.echo("\nFailed files:")
                for r in result['results']:
                    if not r['success']:
                        click.echo(f"  - {r['file_path']}: {r.get('error', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument('directory_path')
@click.option('--output', '-o', help='Output file for results')
def ingest_dir(directory_path, output):
    """Ingest all supported files from a directory."""
    try:
        pipeline = RAGPipeline()
        results = pipeline.ingest_directory(directory_path)
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(f"Directory ingestion completed:")
            click.echo(f"  Directory: {directory_path}")
            click.echo(f"  Total files: {len(results)}")
            click.echo(f"  Successful: {successful}")
            click.echo(f"  Failed: {failed}")
            
            if failed > 0:
                click.echo("\nFailed files:")
                for r in results:
                    if not r['success']:
                        click.echo(f"  - {r['file_path']}: {r.get('error', 'Unknown error')}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument('questions', nargs=-1)
@click.option('--input-file', '-i', help='File containing questions (one per line)')
@click.option('--top-k', default=5, help='Number of top results to retrieve')
@click.option('--output', '-o', help='Output file for results')
def batch_query(questions, input_file, top_k, output):
    """Process multiple queries in batch."""
    try:
        pipeline = RAGPipeline()
        
        # Get questions from file if provided
        if input_file:
            with open(input_file, 'r') as f:
                questions = [line.strip() for line in f if line.strip()]
        
        if not questions:
            click.echo("No questions provided. Use --input-file or provide questions as arguments.", err=True)
            return
        
        results = pipeline.batch_query(questions, top_k)
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(f"Batch query completed for {len(questions)} questions:")
            for i, (question, result) in enumerate(zip(questions, results), 1):
                click.echo(f"\n{i}. Question: {question}")
                click.echo(f"   Success: {result['success']}")
                if result['success']:
                    click.echo(f"   Response: {result['response'][:200]}...")
                    click.echo(f"   Sources: {result['num_sources']}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument('partial_query')
@click.option('--max-suggestions', default=5, help='Maximum number of suggestions')
def suggestions(partial_query, max_suggestions):
    """Get query suggestions based on partial input."""
    try:
        pipeline = RAGPipeline()
        suggestions = pipeline.get_query_suggestions(partial_query, max_suggestions)
        
        click.echo(f"Suggestions for '{partial_query}':")
        for i, suggestion in enumerate(suggestions, 1):
            click.echo(f"{i}. {suggestion}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument('query')
def analyze(query):
    """Analyze query intent and characteristics."""
    try:
        pipeline = RAGPipeline()
        analysis = pipeline.analyze_query(query)
        
        click.echo(f"Analysis for '{query}':")
        for key, value in analysis.items():
            click.echo(f"  {key}: {value}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
def stats():
    """Get pipeline statistics."""
    try:
        pipeline = RAGPipeline()
        stats = pipeline.get_pipeline_stats()
        
        click.echo("Pipeline Statistics:")
        click.echo(json.dumps(stats, indent=2))
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
def health():
    """Check pipeline health."""
    try:
        pipeline = RAGPipeline()
        health_status = pipeline.health_check()
        
        click.echo("Pipeline Health Check:")
        click.echo(f"Overall Status: {health_status['overall_status']}")
        
        for component, status in health_status['components'].items():
            click.echo(f"\n{component.upper()}:")
            for key, value in status.items():
                click.echo(f"  {key}: {value}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument('source_name')
def remove_source(source_name):
    """Remove all documents from a specific source."""
    try:
        pipeline = RAGPipeline()
        success = pipeline.remove_source(source_name)
        
        if success:
            click.echo(f"Successfully removed source: {source_name}")
        else:
            click.echo(f"Failed to remove source: {source_name}", err=True)
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to reset the entire pipeline?')
def reset():
    """Reset the entire pipeline (delete all documents)."""
    try:
        pipeline = RAGPipeline()
        success = pipeline.reset_pipeline()
        
        if success:
            click.echo("Pipeline reset successfully")
        else:
            click.echo("Failed to reset pipeline", err=True)
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == '__main__':
    cli() 