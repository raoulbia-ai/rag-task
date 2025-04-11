"""
Document Indexer Module

This module provides functionality to create searchable indices from processed documents
using vector embeddings for semantic search capabilities.
"""

import os
import json
import numpy as np
import faiss
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# Simple sentence tokenizer function to avoid NLTK dependency issues
def simple_sent_tokenize(text):
    """
    Simple sentence tokenizer that splits text on common sentence endings.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        list: List of sentences
    """
    # Split on common sentence endings (., !, ?)
    # followed by space or newline and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Further split very long sentences (more than 200 chars)
    result = []
    for sentence in sentences:
        if len(sentence) > 200:
            # Split on periods followed by space
            subsents = re.split(r'(?<=\.)\s+', sentence)
            result.extend([s for s in subsents if s])
        else:
            result.append(sentence)
    
    return [s.strip() for s in result if s.strip()]

class DocumentIndexer:
    """Indexer for creating searchable document indices."""
    
    def __init__(self, documents_file):
        """
        Initialize the document indexer.
        
        Args:
            documents_file (str): Path to the JSON file containing processed documents
        """
        self.documents_file = documents_file
        self.documents = []
        self.chunks = []
        self.vectorizer = None
        self.index = None
        self.load_documents()
    
    def load_documents(self):
        """Load processed documents from JSON file."""
        with open(self.documents_file, 'r', encoding='utf-8') as file:
            self.documents = json.load(file)
        print(f"Loaded {len(self.documents)} documents")
    
    def chunk_documents(self, chunk_size=5):
        """
        Split documents into smaller chunks for more precise retrieval.
        
        Args:
            chunk_size (int): Number of sentences per chunk
            
        Returns:
            list: List of document chunks
        """
        self.chunks = []
        chunk_id = 0
        
        for doc in self.documents:
            # Split content into sentences
            sentences = simple_sent_tokenize(doc['content'])
            
            # Create chunks of sentences
            for i in range(0, len(sentences), chunk_size):
                chunk_sentences = sentences[i:i+chunk_size]
                chunk_text = ' '.join(chunk_sentences)
                
                # Create chunk object
                chunk = {
                    'id': chunk_id,
                    'doc_id': doc['id'],
                    'title': doc['title'],
                    'content': chunk_text,
                    'source_file': doc['source_file'],
                    'format': doc['format'],
                    'technology': doc['technology'],
                    'chunk_index': i // chunk_size
                }
                
                self.chunks.append(chunk)
                chunk_id += 1
        
        print(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")
        return self.chunks
    
    def create_tfidf_index(self):
        """
        Create TF-IDF vectorizer and index for document chunks.
        
        Returns:
            tuple: (vectorizer, index, document chunks)
        """
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_df=0.85,
            min_df=2
        )
        
        # Extract chunk texts
        chunk_texts = [chunk['content'] for chunk in self.chunks]
        
        # Fit and transform texts to TF-IDF vectors
        tfidf_vectors = self.vectorizer.fit_transform(chunk_texts)
        
        # Convert sparse matrix to dense numpy array
        dense_vectors = tfidf_vectors.toarray().astype('float32')
        
        # Create FAISS index
        vector_dimension = dense_vectors.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)  # Inner product similarity
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(dense_vectors)
        
        # Add vectors to index
        self.index.add(dense_vectors)
        
        print(f"Created TF-IDF index with {self.index.ntotal} vectors of dimension {vector_dimension}")
        return self.vectorizer, self.index, self.chunks
    
    def search(self, query, top_k=5):
        """
        Search for relevant document chunks using the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of relevant document chunks with scores
        """
        if self.vectorizer is None or self.index is None:
            raise ValueError("Index not created. Call create_tfidf_index() first.")
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        
        # Search index
        scores, indices = self.index.search(query_vector, top_k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):  # Check if index is valid
                chunk = self.chunks[idx]
                results.append({
                    'chunk': chunk,
                    'score': float(scores[0][i])
                })
        
        return results
    
    def save_index(self, output_dir):
        """
        Save the index and related data for later use.
        
        Args:
            output_dir (str): Directory to save index files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunks
        with open(os.path.join(output_dir, 'chunks.json'), 'w', encoding='utf-8') as file:
            json.dump(self.chunks, file, indent=2)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(output_dir, 'document_index.faiss'))
        
        # Save vectorizer vocabulary
        with open(os.path.join(output_dir, 'vectorizer_vocab.json'), 'w', encoding='utf-8') as file:
            json.dump({
                'vocabulary': self.vectorizer.vocabulary_,
                'idf': self.vectorizer.idf_.tolist()
            }, file)
        
        print(f"Saved index and related data to {output_dir}")
    
    @classmethod
    def load_index(cls, index_dir):
        """
        Load previously saved index and related data.
        
        Args:
            index_dir (str): Directory containing index files
            
        Returns:
            DocumentIndexer: Initialized indexer with loaded data
        """
        indexer = cls.__new__(cls)
        
        # Load chunks
        with open(os.path.join(index_dir, 'chunks.json'), 'r', encoding='utf-8') as file:
            indexer.chunks = json.load(file)
        
        # Load FAISS index
        indexer.index = faiss.read_index(os.path.join(index_dir, 'document_index.faiss'))
        
        # Load vectorizer vocabulary
        with open(os.path.join(index_dir, 'vectorizer_vocab.json'), 'r', encoding='utf-8') as file:
            vocab_data = json.load(file)
        
        # Recreate vectorizer
        indexer.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english'
        )
        indexer.vectorizer.vocabulary_ = vocab_data['vocabulary']
        indexer.vectorizer.idf_ = np.array(vocab_data['idf'])
        
        print(f"Loaded index with {indexer.index.ntotal} vectors and {len(indexer.chunks)} chunks")
        return indexer


if __name__ == "__main__":
    # Example usage
    documents_file = "processed_files/extracted_raw_documents.json"
    output_dir = "processed_files/search_index"
    
    indexer = DocumentIndexer(documents_file)
    indexer.chunk_documents()
    indexer.create_tfidf_index()
    indexer.save_index(output_dir)
    
    # Test search
    results = indexer.search("What is the compiler?")
    for result in results:
        print(f"Score: {result['score']:.4f}, Title: {result['chunk']['title']}")
        print(f"Technology: {result['chunk']['technology']}")
        print(f"Source: {result['chunk']['source_file']}")
        print(f"Content: {result['chunk']['content'][:200]}...\n")
