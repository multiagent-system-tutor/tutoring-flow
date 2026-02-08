import os
import json
import faiss
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .utils import setup_logger, time_tracker
from .syllabus_mapper import SyllabusMapper

logger = setup_logger("retriever")

class HybridRetriever:
    def __init__(self, index_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.index_dir = index_dir
        self.mapper = SyllabusMapper() 
        
        meta_path = os.path.join(index_dir, "metadata.json")
        if not os.path.exists(meta_path):
             raise FileNotFoundError(f"Metadata not found at {meta_path}")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        index_path = os.path.join(index_dir, "vector.index")
        if not os.path.exists(index_path):
             raise FileNotFoundError(f"Index not found at {index_path}")
             
        self.index = faiss.read_index(index_path)
        
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self._init_bm25()

    def _init_bm25(self):
        logger.info("Initializing BM25...")
        corpus = [f"{doc['topic']} {doc['question']}" for doc in self.metadata]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search_vector(self, query: str, k: int = 3) -> List[Dict]:
        query_vec = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata) and idx >= 0:
                results.append(self.metadata[idx])
        return results

    def search_keyword(self, query: str, k: int = 3) -> List[Dict]:
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_n_indices:
            results.append(self.metadata[idx])
        return results

    @time_tracker
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        
        vector_docs = self.search_vector(query, k)
        keyword_docs = self.search_keyword(query, k)
        
        seen_ids = set()
        merged_results = []
        
        max_len = max(len(vector_docs), len(keyword_docs))
        for i in range(max_len):
            if i < len(vector_docs):
                doc = vector_docs[i]
                if doc['question_id'] not in seen_ids:
                    merged_results.append(doc)
                    seen_ids.add(doc['question_id'])
            
            if i < len(keyword_docs):
                doc = keyword_docs[i]
                if doc['question_id'] not in seen_ids:
                    merged_results.append(doc)
                    seen_ids.add(doc['question_id'])
        
        return merged_results[:k]

    def format_context(self, retrieved_docs: List[Dict], week_plan: str = None) -> str:
        
        syllabus_context = ""
        if week_plan:
            topic = self.mapper.get_topic_by_week(week_plan)
            syllabus_context = f"## Context: Syllabus (Week {week_plan})\nTopic: {topic}\n\n"
        
        problems_context = "## Context: Reference Problems & Solutions\n" + \
                           "Use these examples to understand the style and difficulty of the questions.\n\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            problems_context += f"### Example {i}\n"
            problems_context += f"**Subject/Topic**: {doc['topic']}\n"
            problems_context += f"**Question**:\n{doc['question']}\n\n"
            problems_context += f"**Solution (Pseudocode)**:\n{doc['answer']}\n"
            problems_context += "---\n"
            
        return syllabus_context + problems_context

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(base_dir, "index")
    
    try:
        retriever = HybridRetriever(index_dir)
        
        query = "Looping dan perulangan"
        print(f"Testing Query: {query}")
        
        docs = retriever.retrieve(query, k=2)
        print(f"\nRetrieved {len(docs)} docs:")
        for d in docs:
            print(f"- {d['topic']}")
            
        print("\n--- Generated Context ---")
        context = retriever.format_context(docs, week_plan="4")
        print(context[:300] + "...\n(truncated)")
    except Exception as e:
        print(f"Error: {e}")
