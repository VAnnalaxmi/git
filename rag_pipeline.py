import os
import shutil
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class RAGPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2", db_path="./chroma_db"):
        """
        Initialize the RAG Pipeline
        
        Args:
            model_name: HuggingFace sentence-transformers model name
            db_path: Path to store ChromaDB data
        """
        self.model_name = model_name
        self.db_path = db_path
        
        # Initialize embedding model (free from HuggingFace)
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        print(f"Initializing ChromaDB at {db_path}...")
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(settings)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
    def load_pdf(self, pdf_path):
        """Extract text from PDF file"""
        print(f"Loading PDF: {pdf_path}...")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            print(f"Successfully extracted {len(pdf_reader.pages)} pages")
            return text
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return ""
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk (characters)
            overlap: Number of overlapping characters between chunks
        """
        print(f"Chunking text (size={chunk_size}, overlap={overlap})...")
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def embed_and_store(self, chunks, metadata=None):
        """
        Generate embeddings and store in ChromaDB
        
        Args:
            chunks: List of text chunks
            metadata: Optional metadata for chunks
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = metadata or [{"source": "pdf", "chunk_id": i} for i in range(len(chunks))]
        
        # Store in ChromaDB
        print("Storing embeddings in ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas
        )
        
        print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
    
    def query(self, query_text, top_k=3):
        """
        Query the RAG system to retrieve relevant documents
        
        Args:
            query_text: Query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks
        """
        print(f"\nQuerying: '{query_text}'")
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return results
    
    def display_results(self, results, top_k=3):
        """Display query results in a readable format"""
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            print("No relevant documents found.")
            return
        
        print(f"\n{'='*80}")
        print(f"Top {top_k} Relevant Documents:")
        print(f"{'='*80}")
        
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"\n[Result {i+1}] (Similarity Score: {1 - distance:.4f})")
            print(f"{'-'*80}")
            print(doc[:500] + "..." if len(doc) > 500 else doc)
            print(f"{'-'*80}")
    
    def load_pdf_and_store(self, pdf_path, chunk_size=500, overlap=50):
        """Complete pipeline: Load PDF → Chunk → Embed → Store"""
        # Load PDF
        text = self.load_pdf(pdf_path)
        if not text:
            return False
        
        # Chunk text
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            print("No chunks created from PDF")
            return False
        
        # Embed and store
        metadata = [{"source": pdf_path, "chunk_id": i} for i in range(len(chunks))]
        self.embed_and_store(chunks, metadata=metadata)
        
        return True
    
    def clear_database(self):
        """Clear the ChromaDB database"""
        print("Clearing ChromaDB...")
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="documents")
        print("Database cleared")
