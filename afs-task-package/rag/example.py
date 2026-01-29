"""
Example usage script demonstrating RAG System capabilities.

Run this script to see the system in action.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.services import RAGService
from src.utils import setup_logging


def main():
    """Example usage of RAG System."""
    print("="*70)
    print("RAG SYSTEM - EXAMPLE USAGE")
    print("="*70)
    
    # 1. Load configuration
    print("\n1. Loading configuration...")
    settings = get_settings("config.yaml")
    print(f"   ✓ Configuration loaded from config.yaml")
    
    # 2. Setup logging
    print("\n2. Setting up logging...")
    setup_logging(
        level="INFO",
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        console=True,
        log_file="logs/example.log"
    )
    print(f"   ✓ Logging configured (logs/example.log)")
    
    # 3. Initialize service
    print("\n3. Initializing RAG Service...")
    service = RAGService(settings)
    print(f"   ✓ RAG Service initialized")
    
    # 4. Build pipeline
    print("\n4. Building pipeline...")
    print("   (This will load existing index if available)")
    service.build_pipeline(force_rebuild=False)
    print(f"   ✓ Pipeline built successfully")
    
    # 5. Get metrics
    print("\n5. System Metrics:")
    metrics = service.get_metrics()
    print(f"   • Total Documents: {metrics.get('total_documents', 0)}")
    print(f"   • Total Chunks: {metrics.get('total_chunks', 0)}")
    print(f"   • Build Time: {metrics.get('pipeline_build_time', 0):.2f}s")
    
    if 'embedding_info' in metrics:
        print(f"   • Embedding Model: {metrics['embedding_info']['model_name']}")
        print(f"   • Embedding Dimension: {metrics['embedding_info']['dimension']}")
    
    # 6. Example queries
    print("\n6. Example Queries:")
    print("-"*70)
    
    # Query 1: Algorithm question
    print("\n   Query 1: Algorithm Implementation")
    result1 = service.query(
        student_input="How do I implement a linear search algorithm to find maximum value?",
        summary="Student asking about algorithm implementation",
        student_profile={"level": "beginner", "course": "Algorithms 101"}
    )
    
    print(f"   Results:")
    print(f"   • Sources Retrieved: {result1['num_sources']}")
    print(f"   • Retrieval Time: {result1['retrieval_time']*1000:.2f}ms")
    print(f"   • Context Preview: {result1['context'][:150]}...")
    
    print("\n   Source Details:")
    for i, source in enumerate(result1['sources'][:2]):
        print(f"     {i+1}. {source['filename']} (score: {source['score']:.4f})")
    
    # Query 2: Exam rules
    print("\n   Query 2: Exam Policy")
    result2 = service.query(
        student_input="Can I use calculator during the exam?",
        summary="Student asking about exam rules",
        student_profile={"level": "intermediate"}
    )
    
    print(f"   Results:")
    print(f"   • Sources Retrieved: {result2['num_sources']}")
    print(f"   • Retrieval Time: {result2['retrieval_time']*1000:.2f}ms")
    print(f"   • Context Preview: {result2['context'][:150]}...")
    
    # 7. Batch processing example
    print("\n7. Batch Processing Example:")
    print("-"*70)
    
    batch_queries = [
        {
            "student_input": "What are arrays?",
            "summary": "Question about data structures"
        },
        {
            "student_input": "Explain for loops",
            "summary": "Question about control flow"
        }
    ]
    
    batch_results = service.batch_query(batch_queries)
    print(f"   Processed {len(batch_results)} queries")
    for i, result in enumerate(batch_results):
        if 'error' not in result:
            print(f"   • Query {i+1}: {result['num_sources']} sources, "
                  f"{result['retrieval_time']*1000:.2f}ms")
    
    # 8. Health check
    print("\n8. System Health Check:")
    print("-"*70)
    health = service.health_check()
    
    for component, status in health.items():
        status_icon = "✓" if status else "✗"
        print(f"   {status_icon} {component}: {'OK' if status else 'FAIL'}")
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext steps:")
    print("  • Try your own queries using: python main.py query \"your question\"")
    print("  • View logs at: logs/example.log")
    print("  • Check README.md for more examples")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
