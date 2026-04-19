from rag_pipeline import RAGPipeline

# Initialize the RAG pipeline
rag = RAGPipeline(model_name="all-MiniLM-L6-v2", db_path="./chroma_db")

# Example 1: Load a PDF and store embeddings
pdf_file = "sample.pdf"  # Replace with your PDF file
if rag.load_pdf_and_store(pdf_file, chunk_size=500, overlap=50):
    print("\n✓ PDF loaded and stored successfully!\n")
    
    # Example 2: Query the stored documents
    queries = [
        "What is the main topic?",
        "Tell me about key concepts",
        "Summarize the document"
    ]
    
    for query in queries:
        results = rag.query(query, top_k=3)
        rag.display_results(results, top_k=3)
else:
    print("Failed to load PDF")

# Example 3: Load multiple PDFs
print("\n" + "="*80)
print("Loading multiple PDFs")
print("="*80)

pdf_files = ["document1.pdf", "document2.pdf"]
for pdf in pdf_files:
    if rag.load_pdf_and_store(pdf, chunk_size=500, overlap=50):
        print(f"✓ {pdf} loaded successfully")
    else:
        print(f"✗ Failed to load {pdf}")

# Query across all documents
print("\n" + "="*80)
print("Querying across all documents")
print("="*80)
results = rag.query("Find information about X", top_k=5)
rag.display_results(results, top_k=5)