import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.src.retriever import HybridRetriever
from rag.src.syllabus_mapper import SyllabusMapper

def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--plan", type=str, required=True, help="Study plan (e.g. 'Minggu 4' or 'Looping')")
    parser.add_argument("--date", type=str, default="", help="Today's date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_dir = os.path.join(base_dir, "index")
    
    if not os.path.exists(index_dir):
        print("Error: Index not found. Please run 'python -m rag.src.ingester' first.")
        sys.exit(1)

    print(f"Loading RAG System for Plan: {args.plan}...")

    mapper = SyllabusMapper()

    topic = mapper.get_topic_by_week(args.plan)
    
    query = topic
    if topic == "General Programming Topic" and "minggu" not in args.plan.lower():
        query = args.plan

    retriever = HybridRetriever(index_dir)
    results = retriever.retrieve(query, k=3)
    
    context_string = retriever.format_context(results, week_plan=args.plan)
    
    output = {
        "input_plan": args.plan,
        "derived_query": query,
        "todays_date": args.date,
        "retrieved_count": len(results),
        "context": context_string
    }
    
    print("\n" + "="*20 + " RESULT " + "="*20)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("="*48)

if __name__ == "__main__":
    main()
