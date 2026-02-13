import os
import json
import time
import statistics
from typing import List, Dict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.src.retriever import HybridRetriever
from rag.src.syllabus_mapper import SyllabusMapper

class RagBenchmark:
    def __init__(self, index_dir: str):
        self.retriever = HybridRetriever(index_dir)
        self.mapper = SyllabusMapper()
        self.results = []

    def run_latency_test(self, query: str, iterations: int = 10) -> Dict:
        latencies = []
        for _ in range(iterations):
            start = time.time()
            _ = self.retriever.retrieve(query, k=5)
            end = time.time()
            latencies.append((end - start) * 1000) # ms
        
        return {
            "query": query,
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))]
        }

    def run_accuracy_heuristic(self, plan: str, expected_keywords: List[str]) -> Dict:
        topic = self.mapper.get_topic_by_week(plan)
        query = topic if topic != "General Programming Topic" else plan
        
        results = self.retriever.retrieve(query, k=5)
        
        hits = 0
        joined_text = " ".join([r['topic'] + " " + r['question'] + " " + r['answer'] for r in results]).lower()
        
        missing_keywords = []
        for kw in expected_keywords:
            if kw.lower() in joined_text:
                hits += 1
            else:
                missing_keywords.append(kw)
        
        score = (hits / len(expected_keywords)) * 100 if expected_keywords else 0
        
        return {
            "plan_input": plan,
            "derived_query": query,
            "relevance_score": score,
            "missing_keywords": missing_keywords,
            "retrieved_count": len(results)
        }

    def run_suite(self):
        print("Running RAG Benchmark Suite...")
        
        latency_queries = ["Looping", "Array", "Function", "Biometrics"]
        accuracy_cases = [
            {"plan": "Minggu 4", "keywords": ["Looping", "For", "While", "Repeat"]},
            {"plan": "Biometrics", "keywords": ["Typing", "Pattern", "Keyboard", "Biometrics"]},
            {"plan": "Card HWSK", "keywords": ["Card", "Poker", "Simbol", "Angka"]}
        ]
        
        suite_results = {
            "latency_metrics": [],
            "accuracy_metrics": []
        }
        
        print("\n--- Latency Tests ---")
        for q in latency_queries:
            res = self.run_latency_test(q)
            print(f"Query: '{q}' -> Avg: {res['avg_latency_ms']:.2f} ms")
            suite_results["latency_metrics"].append(res)
            
        print("\n--- Accuracy Heuristic Tests ---")
        for case in accuracy_cases:
            res = self.run_accuracy_heuristic(case["plan"], case["keywords"])
            print(f"Plan: '{case['plan']}' -> Score: {res['relevance_score']:.1f}%")
            if res['missing_keywords']:
                print(f"  Missing: {res['missing_keywords']}")
            suite_results["accuracy_metrics"].append(res)
            
        output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
        with open(output_path, 'w') as f:
            json.dump(suite_results, f, indent=2)
        
        print(f"\nFull results saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(base_dir, "index")
    
    if os.path.exists(index_dir):
        benchmark = RagBenchmark(index_dir)
        benchmark.run_suite()
    else:
        print("Index not found. Run ingestion first.")
