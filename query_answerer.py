"""
Query Answering System

This module provides functionality to answer user questions using the document index,
generate comprehensive responses, and include proper citations.
"""

import os
import json
import re
from pathlib import Path
from document_indexer import DocumentIndexer

class QueryAnswerer:
    """System for answering user queries with citations."""
    
    def __init__(self, index_dir):
        """
        Initialize the query answerer.
        
        Args:
            index_dir (str): Directory containing the document index
        """
        self.index_dir = index_dir
        self.indexer = DocumentIndexer.load_index(index_dir)
        
        # Load chunks for citation information
        with open(os.path.join(index_dir, 'chunks.json'), 'r', encoding='utf-8') as file:
            self.chunks = json.load(file)
    
    def answer_question(self, question, top_k=10, max_context_chunks=5):
        """
        Answer a user question using the document index.
        
        Args:
            question (str): User question
            top_k (int): Number of top results to retrieve
            max_context_chunks (int): Maximum number of chunks to include in context
            
        Returns:
            dict: Answer with citations
        """
        # Search for relevant chunks
        search_results = self.indexer.search(question, top_k=top_k)
        
        # Filter results by relevance score threshold
        relevant_results = [r for r in search_results if r['score'] > 0.2]
        
        # Limit to max_context_chunks
        context_results = relevant_results[:max_context_chunks]
        
        if not context_results:
            return {
                'question': question,
                'answer': "I couldn't find relevant information to answer this question in the provided documentation.",
                'citations': []
            }
        
        # Build context from relevant chunks
        context = self._build_context(context_results)
        
        # Generate answer
        answer = self._generate_answer(question, context_results)
        
        # Create citations
        citations = self._create_citations(context_results)
        
        return {
            'question': question,
            'answer': answer,
            'citations': citations
        }
    
    def _build_context(self, results):
        """
        Build context from search results.
        
        Args:
            results (list): Search results
            
        Returns:
            str: Combined context
        """
        context_parts = []
        
        for result in results:
            chunk = result['chunk']
            context_parts.append(f"Source: {chunk['source_file']}\nContent: {chunk['content']}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question, results):
        """
        Generate an answer based on the question and search results.
        
        Args:
            question (str): User question
            results (list): Search results
            
        Returns:
            str: Generated answer
        """
        # In a real implementation, this would use an LLM to generate the answer
        # For this implementation, we'll create a structured answer from the top results
        
        # Extract technology from top result
        top_technology = results[0]['chunk']['technology'] if results else "unknown"
        
        # Create answer introduction based on question type
        if re.search(r'what is|what are|explain|describe', question.lower()):
            intro = f"Based on the {top_technology} documentation, "
        elif re.search(r'how to|how do|steps|process', question.lower()):
            intro = f"According to the {top_technology} documentation, you can "
        elif re.search(r'is|are|can|does|do', question.lower()):
            intro = f"The {top_technology} documentation indicates that "
        else:
            intro = f"Based on the {top_technology} documentation, "
        
        # Combine content from top results
        content_parts = []
        for i, result in enumerate(results):
            chunk = result['chunk']
            # Add citation marker
            content = f"{chunk['content']} [citation:{i+1}]"
            content_parts.append(content)
        
        # Join content parts with appropriate transitions
        if len(content_parts) == 1:
            content = content_parts[0]
        else:
            # Use different joining strategies based on question type
            if re.search(r'how to|steps|process', question.lower()):
                # For how-to questions, number the steps
                numbered_parts = [f"{i+1}. {part}" for i, part in enumerate(content_parts)]
                content = "\n".join(numbered_parts)
            else:
                # For other questions, join with transitions
                transitions = [
                    "Additionally, ", 
                    "Furthermore, ", 
                    "Moreover, ", 
                    "Also, ", 
                    "In addition, "
                ]
                joined_parts = [content_parts[0]]
                for i, part in enumerate(content_parts[1:]):
                    transition = transitions[i % len(transitions)]
                    joined_parts.append(f"{transition}{part}")
                content = " ".join(joined_parts)
        
        # Combine intro and content
        answer = f"{intro}{content}"
        
        # Clean up citation markers for better readability
        answer = re.sub(r'\s+\[citation:(\d+)\]', r' [citation:\1]', answer)
        
        return answer
    
    def _create_citations(self, results):
        """
        Create citations from search results.
        
        Args:
            results (list): Search results
            
        Returns:
            list: Citations
        """
        citations = []
        
        for i, result in enumerate(results):
            chunk = result['chunk']
            citation = {
                'id': i + 1,
                'technology': chunk['technology'],
                'source_file': chunk['source_file'],
                'title': chunk['title'],
                'relevance': result['score']
            }
            citations.append(citation)
        
        return citations


if __name__ == "__main__":
    # Example usage
    index_dir = "processed_files/search_index"
    answerer = QueryAnswerer(index_dir)
    
    # Test with example questions
    example_questions = [
        "What is the compiler?",
        "How do I get started developing a UI?",
        "How to secure my cluster?",
        "How to update the SSL keystore?",
        "Is streaming supported?"
    ]
    
    for question in example_questions:
        print(f"\nQuestion: {question}")
        answer_data = answerer.answer_question(question)
        print(f"Answer: {answer_data['answer']}")
        print("Citations:")
        for citation in answer_data['citations']:
            print(f"  [{citation['id']}] {citation['title']} ({citation['source_file']})")
