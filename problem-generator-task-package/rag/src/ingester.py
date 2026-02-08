import os
import json
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from .utils import setup_logger, time_tracker

logger = setup_logger("ingester")

class RagIngester:
    def __init__(self, processed_dir: str, index_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.processed_dir = processed_dir
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def load_data(self) -> List[Dict]:
        data = []
        file_path = os.path.join(self.processed_dir, "dataset.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} documents from {file_path}")
        return data

    def create_text_representation(self, entry: Dict) -> str:
        return f"Topic: {entry['topic']}. Question: {entry['question']}"

    @time_tracker
    def build_index(self):
        data = self.load_data()
        if not data:
            logger.warning("No data to ingest.")
            return

        texts = [self.create_text_representation(d) for d in data]
        
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        embeddings = np.array(embeddings).astype('float32')

        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings)
        
        logger.info(f"Index built with {index.ntotal} vectors.")
        
        index_path = os.path.join(self.index_dir, "vector.index")
        faiss.write_index(index, index_path)
        logger.info(f"FAISS index saved to {index_path}")
        
        metadata_path = os.path.join(self.index_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "dataset", "processed")
    index_dir = os.path.join(base_dir, "index")
    
    ingester = RagIngester(processed_dir, index_dir)
    ingester.build_index()
