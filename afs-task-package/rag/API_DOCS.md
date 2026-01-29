# API Documentation

## RAG Service API Reference

This document provides detailed API documentation for the RAG System.

## Table of Contents

- [Configuration](#configuration)
- [Core Modules](#core-modules)
- [Service API](#service-api)
- [CLI Commands](#cli-commands)
- [Python API Examples](#python-api-examples)

---

## Configuration

### Settings Class

Located in `src/config/settings.py`

```python
from src.config import get_settings

# Load settings
settings = get_settings("config.yaml")

# Access configuration
print(settings.embedding.model_name)
print(settings.chunking.chunk_size)
print(settings.retrieval.top_k)
```

#### Configuration Sections

1. **DataConfig**
   - `data_dir`: Directory containing documents
   - `supported_formats`: List of file extensions
   - `encoding`: File encoding

2. **ChunkingConfig**
   - `chunk_size`: Maximum chunk size in characters
   - `chunk_overlap`: Overlap between chunks
   - `separators`: List of separators for splitting

3. **EmbeddingConfig**
   - `model_name`: HuggingFace model name
   - `device`: 'cpu' or 'cuda'
   - `normalize_embeddings`: Boolean
   - `batch_size`: Batch size for processing

4. **VectorStoreConfig**
   - `type`: Vector store type ('faiss')
   - `index_path`: Path to save/load index
   - `save_on_create`: Auto-save after creation

5. **RetrievalConfig**
   - `top_k`: Number of documents to retrieve
   - `score_threshold`: Minimum similarity score
   - `return_scores`: Include similarity scores

---

## Core Modules

### DataLoader

**Purpose**: Load documents from file system

```python
from src.core import DataLoader

loader = DataLoader(
    data_dir="raw_data",
    supported_formats=[".txt"],
    encoding="utf-8"
)

documents = loader.load_documents()
stats = loader.get_statistics()
```

**Methods**:

- `load_documents() -> List[Document]`: Load all documents
- `get_statistics() -> dict`: Get loading statistics

### TextPreprocessor

**Purpose**: Clean and normalize text

```python
from src.core import TextPreprocessor

preprocessor = TextPreprocessor(
    remove_special_chars=True,
    normalize_whitespace=True,
    preserve_case=True
)

processed_docs = preprocessor.preprocess_documents(documents)
```

**Methods**:

- `preprocess_documents(docs) -> List[Document]`: Preprocess documents
- `get_statistics(original, processed) -> dict`: Get preprocessing stats

### DocumentChunker

**Purpose**: Split documents into chunks

```python
from src.core import DocumentChunker

chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

chunks = chunker.chunk_documents(documents)
```

**Methods**:

- `chunk_documents(docs) -> List[Document]`: Split documents
- `get_statistics(chunks) -> dict`: Get chunk statistics
- `validate_chunks(chunks) -> bool`: Validate chunk sizes

### EmbeddingGenerator

**Purpose**: Generate vector embeddings

```python
from src.core import EmbeddingGenerator

generator = EmbeddingGenerator(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True
)

# Single query
embedding = generator.embed_query("What is Python?")

# Multiple documents
embeddings = generator.embed_documents(["text1", "text2"])

# Test embedding
test_results = generator.test_embedding("test text")
```

**Methods**:

- `embed_query(text) -> List[float]`: Embed single query
- `embed_documents(texts) -> List[List[float]]`: Embed multiple texts
- `test_embedding(text) -> dict`: Test embedding generation
- `get_model_info() -> dict`: Get model information

### VectorStoreManager

**Purpose**: Manage FAISS vector store

```python
from src.core import VectorStoreManager

manager = VectorStoreManager(
    embeddings=generator.embeddings,
    index_path="faiss_index",
    save_on_create=True
)

# Create index
vector_store = manager.create_index(chunks)

# Save index
manager.save_index()

# Load index
vector_store = manager.load_index()

# Check if exists
if manager.index_exists():
    print("Index found")
```

**Methods**:

- `create_index(docs) -> FAISS`: Create FAISS index
- `save_index(path) -> None`: Save index to disk
- `load_index(path) -> FAISS`: Load index from disk
- `index_exists(path) -> bool`: Check if index exists
- `get_vector_store() -> FAISS`: Get current vector store
- `get_statistics() -> dict`: Get vector store stats

### Retriever

**Purpose**: Perform similarity search

```python
from src.core import Retriever

retriever = Retriever(
    vector_store=vector_store,
    top_k=3,
    score_threshold=0.0,
    return_scores=True
)

# Retrieve documents
docs, time = retriever.retrieve("query text")

# Retrieve with scores
results, time = retriever.retrieve_with_scores("query text")

# Process complete query
output = retriever.process_query(
    student_input="What is Python?",
    summary="Question about programming language",
    student_profile={"level": "beginner"}
)
```

**Methods**:

- `retrieve(query, k) -> Tuple[List[Document], float]`: Retrieve documents
- `retrieve_with_scores(query, k) -> Tuple[List[Tuple[Document, float]], float]`: Retrieve with scores
- `format_context(results) -> str`: Format results as context string
- `process_query(student_input, summary, profile) -> dict`: Process complete query
- `get_statistics(results) -> dict`: Get retrieval statistics

---

## Service API

### RAGService

**Purpose**: High-level RAG pipeline API

```python
from src.config import get_settings
from src.services import RAGService

# Initialize
settings = get_settings()
service = RAGService(settings)

# Build pipeline
service.build_pipeline(force_rebuild=False)

# Query
result = service.query(
    student_input="How do I implement sorting?",
    summary="Question about algorithms",
    student_profile={"level": "beginner"},
    top_k=3
)

# Batch query
queries = [
    {"student_input": "What are arrays?"},
    {"student_input": "Explain loops"}
]
results = service.batch_query(queries)

# Get metrics
metrics = service.get_metrics()

# Health check
health = service.health_check()

# Rebuild index
service.rebuild_index()
```

**Methods**:

#### `build_pipeline(force_rebuild: bool = False) -> None`

Build the complete RAG pipeline. Loads existing index if available, unless `force_rebuild=True`.

**Raises**: `RAGServiceError` if build fails

#### `query(student_input: str, summary: str = "", student_profile: dict = None, top_k: int = None) -> dict`

Process a student query and retrieve relevant context.

**Parameters**:

- `student_input`: Student's answer or question (required)
- `summary`: Summary of student's work (optional)
- `student_profile`: Student information (optional)
- `top_k`: Number of documents to retrieve (optional)

**Returns**: Dictionary with keys:

- `context`: Retrieved context text
- `student_input`: Original input
- `summary`: Original summary
- `student_profile`: Original profile
- `retrieval_time`: Time taken for retrieval
- `num_sources`: Number of sources retrieved
- `sources`: List of source dictionaries with metadata

**Raises**: `RAGServiceError` if query fails

#### `batch_query(queries: List[dict]) -> List[dict]`

Process multiple queries in batch.

**Parameters**:

- `queries`: List of query dictionaries (each with `student_input`, optional `summary`, `student_profile`, `top_k`)

**Returns**: List of result dictionaries

#### `get_metrics() -> dict`

Get pipeline metrics and statistics.

**Returns**: Dictionary containing:

- `total_documents`: Number of documents loaded
- `total_chunks`: Number of chunks created
- `pipeline_build_time`: Time to build pipeline
- `data_stats`: Data loading statistics
- `embedding_info`: Embedding model information
- `vector_store_stats`: Vector store statistics

#### `health_check() -> dict`

Perform health check on all components.

**Returns**: Dictionary with boolean status for each component:

- `initialized`: Service initialization status
- `data_loader`: Data loader status
- `preprocessor`: Preprocessor status
- `chunker`: Chunker status
- `embedding_generator`: Embedding generator status
- `vector_store`: Vector store status
- `retriever`: Retriever status
- `overall`: Overall health status

#### `rebuild_index() -> None`

Force rebuild the vector index from scratch.

---

## CLI Commands

### Build Pipeline

```bash
python main.py build [--force-rebuild] [--config config.yaml] [--no-log-file]
```

**Options**:

- `--force-rebuild`: Force rebuild even if index exists
- `--config`: Path to configuration file (default: config.yaml)
- `--no-log-file`: Disable file logging

**Example**:

```bash
python main.py build --force-rebuild
```

### Query System

```bash
python main.py query "student input" [--summary "text"] [--level {beginner,intermediate,advanced}] [--top-k N]
```

**Options**:

- `--summary`: Summary of student work
- `--level`: Student level (beginner, intermediate, advanced)
- `--top-k`: Number of documents to retrieve

**Example**:

```bash
python main.py query "What is Python?" --summary "Question about programming" --level beginner --top-k 3
```

### Health Check

```bash
python main.py health
```

Returns exit code 0 if healthy, 1 if unhealthy.

### View Metrics

```bash
python main.py metrics
```

Displays system metrics and statistics.

---

## Python API Examples

### Complete Example

```python
from src.config import get_settings
from src.services import RAGService
from src.utils import setup_logging

# Setup
settings = get_settings("config.yaml")
setup_logging(level="INFO", console=True)

# Initialize service
service = RAGService(settings)
service.build_pipeline()

# Simple query
result = service.query(
    student_input="How do I implement bubble sort?"
)

print(f"Context: {result['context']}")
print(f"Sources: {result['num_sources']}")

# Query with full context
result = service.query(
    student_input="Explain recursion",
    summary="Student asking about advanced concepts",
    student_profile={
        "level": "intermediate",
        "course": "CS102",
        "previous_topics": ["loops", "functions"]
    },
    top_k=5
)

# Access detailed results
for source in result['sources']:
    print(f"File: {source['filename']}")
    print(f"Score: {source['score']:.4f}")
    print(f"Preview: {source['preview']}")
```

### Batch Processing

```python
# Prepare batch queries
queries = [
    {
        "student_input": "What are variables?",
        "summary": "Basic programming question",
        "student_profile": {"level": "beginner"}
    },
    {
        "student_input": "Explain classes and objects",
        "summary": "OOP question",
        "student_profile": {"level": "intermediate"}
    },
    {
        "student_input": "What is polymorphism?",
        "summary": "Advanced OOP question",
        "student_profile": {"level": "advanced"},
        "top_k": 5
    }
]

# Process batch
results = service.batch_query(queries)

# Process results
for i, result in enumerate(results):
    if 'error' not in result:
        print(f"Query {i+1}: {result['num_sources']} sources")
    else:
        print(f"Query {i+1}: Error - {result['error']}")
```

### Custom Configuration

```python
from src.config import Settings, ChunkingConfig, RetrievalConfig

# Create custom settings
settings = Settings(
    chunking=ChunkingConfig(
        chunk_size=1000,
        chunk_overlap=100
    ),
    retrieval=RetrievalConfig(
        top_k=5,
        score_threshold=0.5
    )
)

service = RAGService(settings)
service.build_pipeline()
```

### Error Handling

```python
from src.services import RAGServiceError

try:
    service = RAGService(settings)
    service.build_pipeline()

    result = service.query("test query")

except RAGServiceError as e:
    print(f"RAG Service Error: {str(e)}")
    # Handle specific error

except Exception as e:
    print(f"Unexpected Error: {str(e)}")
    # Handle unexpected errors
```

---

## Output Format

All query methods return a dictionary with the following structure:

```python
{
    "context": str,              # Retrieved context text
    "student_input": str,        # Original student input
    "summary": str,              # Original summary
    "student_profile": dict,     # Original student profile
    "retrieval_time": float,     # Retrieval time in seconds
    "num_sources": int,          # Number of sources retrieved
    "sources": [                 # List of source documents
        {
            "filename": str,     # Source file name
            "category": str,     # Source category
            "score": float,      # Similarity score
            "preview": str       # First 150 characters
        },
        ...
    ]
}
```

---

## Performance Tips

1. **Use GPU for faster embedding** (if available):

   ```yaml
   embedding:
     device: 'cuda'
   ```

2. **Adjust chunk size** for better retrieval:

   ```yaml
   chunking:
     chunk_size: 1000 # Larger chunks = more context
     chunk_overlap: 100
   ```

3. **Increase top_k** for more comprehensive results:

   ```yaml
   retrieval:
     top_k: 5
   ```

4. **Enable caching** in production:

   ```yaml
   system:
     enable_caching: true
   ```

5. **Use batch processing** for multiple queries:
   ```python
   results = service.batch_query(queries)
   ```

---

## Support

For issues and questions:

- Check the [README](README.md)
- Review logs in `logs/rag_system.log`
- Run health check: `python main.py health`
- Enable DEBUG logging for detailed information
