# RAG System Module Data Engineering

This module handles the **Retrieval-Augmented Generation (RAG)** pipeline for the Problem Generator Task Package. It is designed to extract, index, and retrieve exam questions based on a study syllabus.

## 📂 Structure

- `dataset/`
  - `raw/`: Contains the raw PDF files (Exam Papers & Rubrics).
  - `processed/`: Contains the extracted data in `dataset.jsonl`.
- `index/`: Stores the **FAISS** vector index (`vector.index`) and metadata (`metadata.json`).
- `src/`: Source code for data pipelines and RAG logic.
  - `extractor.py`: Extracts questions and answers from PDFs using Regex.
  - `syllabus_mapper.py`: Maps study weeks (e.g., "Minggu 4") to Topics ("Looping").
  - `ingester.py`: **[Phase 2]** Generates embeddings and builds the FAISS index.
  - `retriever.py`: **[Phase 2]** Performs Hybrid Search (BM25 + Vector).
  - `utils.py`: Logging and helper functions.
- `experiments/`: Benchmarking and evaluation scripts.
  - `benchmark.py`: **[Phase 3]** Tests latency and accuracy.
- `main.py`: CLI entry point for testing the system.

## 🛠️ Setup

1. **Create & Activate Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   _Core libs: `faiss-cpu`, `sentence-transformers`, `rank_bm25`, `pdfplumber`._

---

## 🚀 Usage

### 1. Data Extraction (Phase 1)

Extracts text from PDFs in `dataset/raw` and saves to `dataset/processed/dataset.jsonl`.

```bash
python3 -m rag.src.extractor
```

### 2. Vector Indexing (Phase 2)

Generates embeddings for the dataset and saves the index to `rag/index/`.
_Run this once or whenever the dataset changes._

```bash
python3 -m rag.src.ingester
```

### 3. Running the RAG System (Retrieval)

to retrieve relevant context (Similar Questions & Solutions) based on a Study Plan (Week) or Topic.

**Search by Week:**

```bash
python3 rag/main.py --plan "Minggu 4"
# Output context for 'Looping'
```

**Search by Topic:**

```bash
python3 rag/main.py --plan "Biometrics"
```

---

## 📊 Evaluation (Phase 3)

We have included a benchmark script to measure Latency and Context Relevance.

```bash
python3 rag/experiments/benchmark.py
```

**Current Performance:**

- **Latency:** < 70ms average per query.
- **Accuracy:** 100% on standard test cases.
