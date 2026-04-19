# Basic RAG Pipeline

A simple, free Retrieval-Augmented Generation (RAG) system using HuggingFace embeddings and ChromaDB.

## 📚 Project Overview

This project implements a basic RAG pipeline that:
- Loads PDF documents
- Chunks text intelligently with overlap
- Generates embeddings using free HuggingFace models
- Stores embeddings in ChromaDB (local vector database)
- Retrieves and displays relevant documents based on queries

## 🚀 Features

✅ **Free & Open Source** - Uses HuggingFace and ChromaDB (no paid APIs)
✅ **Local Storage** - All data stored locally, no cloud dependency
✅ **PDF Support** - Automatically extracts and processes PDF documents
✅ **Semantic Search** - Finds relevant content using embeddings
✅ **Easy Integration** - Simple Python API

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/VAnnalaxmi/git.git
cd git
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Example

```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(model_name="all-MiniLM-L6-v2", db_path="./chroma_db")

# Load and store a PDF
rag.load_pdf_and_store("document.pdf", chunk_size=500, overlap=50)

# Query
results = rag.query("What is the main topic?", top_k=3)
rag.display_results(results)
```

### Running the Example

```bash
python example_usage.py
```

## 📁 Project Structure

```
├── requirements.txt       # Python dependencies
├── rag_pipeline.py        # Main RAG implementation
├── example_usage.py       # Example usage script
└── README.md              # This file
```

## 🔧 API Reference

### RAGPipeline Class

#### `__init__(model_name, db_path)`
Initialize the RAG pipeline.

**Parameters:**
- `model_name` (str): HuggingFace sentence-transformers model name (default: "all-MiniLM-L6-v2")
- `db_path` (str): Path to store ChromaDB data (default: "./chroma_db")

#### `load_pdf(pdf_path)`
Extract text from a PDF file.

#### `chunk_text(text, chunk_size, overlap)`
Split text into chunks with overlap.

#### `embed_and_store(chunks, metadata)`
Generate embeddings and store in ChromaDB.

#### `query(query_text, top_k)`
Search for relevant documents.

**Parameters:**
- `query_text` (str): Your search query
- `top_k` (int): Number of top results to return (default: 3)

#### `display_results(results, top_k)`
Display query results in a readable format.

#### `load_pdf_and_store(pdf_path, chunk_size, overlap)`
Complete pipeline: Load PDF → Chunk → Embed → Store

#### `clear_database()`
Clear all stored embeddings.

## 🎯 Supported Models

The pipeline uses HuggingFace sentence-transformers. Some popular models:
- `all-MiniLM-L6-v2` (Fast, lightweight)
- `all-mpnet-base-v2` (Higher quality)
- `distiluse-base-multilingual-cased-v2` (Multilingual)

## 📊 How It Works

1. **PDF Loading** → Extract text using PyPDF2
2. **Chunking** → Split into overlapping chunks for context
3. **Embedding** → Generate vectors using HuggingFace
4. **Storage** → Store in ChromaDB with metadata
5. **Querying** → Embed query and find similar documents
6. **Retrieval** → Return top-k most relevant results

## 🛠️ Requirements

See `requirements.txt` for all dependencies:
- PyPDF2 - PDF text extraction
- sentence-transformers - Free embeddings
- chromadb - Vector database
- langchain - LLM utilities

## 📝 Notes

- First run will download the embedding model (~100MB)
- ChromaDB data is persisted locally in `./chroma_db/`
- Supports multiple PDFs in the same database
- All processing happens locally - no data sent to external services

## 🤝 Contributing

Feel free to extend this project with:
- Different embedding models
- Answer generation from retrieved documents
- Web UI
- API server
- Support for other document formats

## 📄 License

MIT License - Feel free to use for any purpose

## ❓ Support

For issues and questions, create a GitHub issue in this repository.

---

**Happy RAG-ing!** 🚀