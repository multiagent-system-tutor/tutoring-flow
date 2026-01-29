"""
Main execution script for RAG System.

This script provides a command-line interface for running the RAG system,
including pipeline building, querying, and system management.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.services import RAGService
from src.utils import setup_logging


def build_pipeline(args: argparse.Namespace) -> None:
    """Build the RAG pipeline."""
    print("Building RAG pipeline...")
    
    # Load settings
    settings = get_settings(args.config)
    
    # Setup logging
    setup_logging(
        level=settings.logging.level,
        log_format=settings.logging.format,
        log_file=settings.logging.file if not args.no_log_file else None,
        console=settings.logging.console
    )
    
    # Initialize service
    service = RAGService(settings)
    
    # Build pipeline
    service.build_pipeline(force_rebuild=args.force_rebuild)
    
    # Print metrics
    metrics = service.get_metrics()
    print("\n" + "="*60)
    print("PIPELINE BUILT SUCCESSFULLY")
    print("="*60)
    print(f"Total Documents: {metrics.get('total_documents', 0)}")
    print(f"Total Chunks: {metrics.get('total_chunks', 0)}")
    print(f"Build Time: {metrics.get('pipeline_build_time', 0):.2f}s")
    print("="*60)


def query_pipeline(args: argparse.Namespace) -> None:
    """Query the RAG pipeline."""
    # Load settings
    settings = get_settings(args.config)
    
    # Setup logging
    setup_logging(
        level=settings.logging.level,
        log_format=settings.logging.format,
        log_file=settings.logging.file if not args.no_log_file else None,
        console=settings.logging.console
    )
    
    # Initialize service
    service = RAGService(settings)
    
    # Build pipeline (will load existing if available)
    service.build_pipeline(force_rebuild=False)
    
    # Process query
    result = service.query(
        student_input=args.input,
        summary=args.summary or "",
        student_profile={"level": args.level} if args.level else None,
        top_k=args.top_k
    )
    
    # Print results
    print("\n" + "="*60)
    print("QUERY RESULTS")
    print("="*60)
    print(f"Student Input: {result['student_input'][:100]}...")
    print(f"Sources Retrieved: {result['num_sources']}")
    print(f"Retrieval Time: {result['retrieval_time']*1000:.2f}ms")
    print("\n" + "-"*60)
    print("CONTEXT:")
    print("-"*60)
    print(result['context'][:500] + "..." if len(result['context']) > 500 else result['context'])
    print("\n" + "-"*60)
    print("SOURCES:")
    for i, source in enumerate(result['sources']):
        print(f"\n{i+1}. {source['filename']} (score: {source['score']:.4f})")
        print(f"   {source['preview']}...")
    print("="*60)


def health_check(args: argparse.Namespace) -> None:
    """Perform health check on the system."""
    # Load settings
    settings = get_settings(args.config)
    
    # Setup logging (minimal for health check)
    setup_logging(level="WARNING", console=True, log_file=None)
    
    # Initialize service
    service = RAGService(settings)
    
    try:
        # Try to load existing pipeline
        service.build_pipeline(force_rebuild=False)
        
        # Perform health check
        health = service.health_check()
        
        print("\n" + "="*60)
        print("HEALTH CHECK")
        print("="*60)
        for component, status in health.items():
            status_icon = "✓" if status else "✗"
            print(f"{status_icon} {component}: {'OK' if status else 'FAIL'}")
        print("="*60)
        
        sys.exit(0 if health['overall'] else 1)
        
    except Exception as e:
        print(f"\n✗ Health check failed: {str(e)}")
        sys.exit(1)


def get_metrics(args: argparse.Namespace) -> None:
    """Get system metrics."""
    # Load settings
    settings = get_settings(args.config)
    
    # Setup logging (minimal)
    setup_logging(level="WARNING", console=True, log_file=None)
    
    # Initialize service
    service = RAGService(settings)
    service.build_pipeline(force_rebuild=False)
    
    # Get metrics
    metrics = service.get_metrics()
    
    print("\n" + "="*60)
    print("SYSTEM METRICS")
    print("="*60)
    print(f"Pipeline Build Time: {metrics.get('pipeline_build_time', 0):.2f}s")
    print(f"Total Documents: {metrics.get('total_documents', 0)}")
    print(f"Total Chunks: {metrics.get('total_chunks', 0)}")
    
    if 'embedding_info' in metrics:
        print("\nEmbedding Info:")
        for key, value in metrics['embedding_info'].items():
            print(f"  {key}: {value}")
    
    if 'vector_store_stats' in metrics:
        print("\nVector Store:")
        for key, value in metrics['vector_store_stats'].items():
            print(f"  {key}: {value}")
    
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System for Teacher Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--no-log-file',
        action='store_true',
        help='Disable file logging'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build RAG pipeline')
    build_parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild even if index exists'
    )
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument(
        'input',
        type=str,
        help='Student input/question'
    )
    query_parser.add_argument(
        '--summary',
        type=str,
        help='Summary of student work'
    )
    query_parser.add_argument(
        '--level',
        type=str,
        choices=['beginner', 'intermediate', 'advanced'],
        help='Student level'
    )
    query_parser.add_argument(
        '--top-k',
        type=int,
        help='Number of documents to retrieve'
    )
    
    # Health check command
    subparsers.add_parser('health', help='Perform system health check')
    
    # Metrics command
    subparsers.add_parser('metrics', help='Get system metrics')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'build':
        build_pipeline(args)
    elif args.command == 'query':
        query_pipeline(args)
    elif args.command == 'health':
        health_check(args)
    elif args.command == 'metrics':
        get_metrics(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
