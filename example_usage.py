# Example Usage of RAG Pipeline

# Import necessary libraries
from rag_pipeline import load_pdf, chunk_text, embed_text, store_in_chromadb, query_chromadb

# Load a PDF document
pdf_path = 'path/to/your/document.pdf'
pdf_document = load_pdf(pdf_path)

# Chunk the loaded document into manageable parts
chunks = chunk_text(pdf_document)

# Embed the chunks using a suitable embedding function
embedded_chunks = [embed_text(chunk) for chunk in chunks]

# Store the embedded chunks in ChromaDB
store_in_chromadb(embedded_chunks)

# Query the database with a sample query
sample_query = 'What is the main topic of the PDF?'
query_result = query_chromadb(sample_query)

# Display results
print('Query Results:')
for result in query_result:
    print(result) 
