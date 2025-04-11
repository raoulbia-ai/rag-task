"""
Main Document Processing Script

This script orchestrates the document parsing and indexing process,
processing all documentation files and creating a searchable index.
"""

import os
import sys
import time
from document_parser import DocumentParser
from document_indexer import DocumentIndexer

def main():
    """Main function to process documents and create index."""
    print("Starting document processing...")
    start_time = time.time()
    
    # Define paths
    base_dir = "files/"
    output_dir = "processed_files"
    documents_file = os.path.join(output_dir, "extracted_raw_documents.json")
    index_dir = os.path.join(output_dir, "search_index")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    
    # Step 1: Parse documents
    print("\n=== Parsing Documents ===")
    parser = DocumentParser(base_dir)
    documents = parser.process_directory()
    parser.save_documents(documents_file)
    
    # Step 2: Create document index
    print("\n=== Creating Document Index ===")
    indexer = DocumentIndexer(documents_file)
    indexer.chunk_documents(chunk_size=5)
    indexer.create_tfidf_index()
    indexer.save_index(index_dir)
    
    # Step 3: Test search functionality
    print("\n=== Testing Search Functionality ===")
    test_queries = [
        "What is the compiler?",
        "How do I get started developing a UI?",
        "How to secure my cluster?",
        "How to update the SSL keystore?",
        "Is streaming supported?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = indexer.search(query, top_k=3)
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Technology: {result['chunk']['technology']}")
            print(f"  Source: {result['chunk']['source_file']}")
            print(f"  Title: {result['chunk']['title']}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nDocument processing completed in {elapsed_time:.2f} seconds")
    print(f"Processed {len(documents)} documents")
    print(f"Created {len(indexer.chunks)} document chunks")
    print(f"Index saved to {index_dir}")

if __name__ == "__main__":
    main()
