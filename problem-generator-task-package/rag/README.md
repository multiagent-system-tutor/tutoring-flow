# RAG System Module

This module handles the Retrieval-Augmented Generation pipeline for the Problem Generator Task Package.

## Structure

- `dataset/raw`: Contains the raw PDF files (Exam Papers & Rubrics).
- `dataset/processed`: Contains the extracted data in JSONL format.
- `src/`: Source code for data engineering and RAG logic.
  - `extractor.py`: Extracts questions and answers from PDFs.
  - `syllabus_mapper.py`: Maps study weeks to topics.
  - `utils.py`: Logging and helper functions.
- `experiments/`: Log of experiments (future use).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Extraction (Phase 1)

To extract data from the PDFs in `dataset/raw` to `dataset/processed/dataset.jsonl`:

```bash
python3 -m rag.src.extractor
```

### Syllabus Mapping

To query topics by week:

```python
from rag.src.syllabus_mapper import SyllabusMapper
mapper = SyllabusMapper()
print(mapper.get_topic_by_week("1"))
```
