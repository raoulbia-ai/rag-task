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
from openai import OpenAI

class QueryAnswerer:
    """System for answering user queries with citations."""
    
    def __init__(self, index_dir, use_llm=True):
        """
        Initialize the query answerer.
        
        Args:
            index_dir (str): Directory containing the document index
            use_llm (bool): Whether to use LLM for answer generation
        """
        self.index_dir = index_dir
        self.indexer = DocumentIndexer.load_index(index_dir)
        self.use_llm = use_llm and os.environ.get("OPENAI_API_KEY") is not None
        
        # Initialize OpenAI client if API key is available
        if self.use_llm:
            try:
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                self.use_llm = False
        
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
        
        # Generate answer using LLM (even if no relevant results found)
        try:
            answer = self._generate_llm_answer(question, context_results)
        except Exception as e:
            # Handle errors with informative messages
            error_message = f"Error generating answer: {str(e)}"
            print(error_message)
            return {
                'question': question,
                'answer': "Sorry, I encountered an error while generating your answer. Please check the API key or try again later.",
                'citations': [],
                'error': error_message
            }
        
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
    
    def _generate_llm_answer(self, question, results, max_tokens=1024):
        """
        Generate an answer using an LLM based on the question and search results.
        
        Args:
            question (str): User question
            results (list): Search results
            max_tokens (int): Maximum length of generated response
            
        Returns:
            str: Generated answer with citations
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(results):
            chunk = result['chunk']
            # Add source information for citation
            context_part = f"Document {i+1}: {chunk['title']} (from {chunk['technology']} documentation)\n"
            context_part += f"Content: {chunk['content']}\n"
            context_parts.append(context_part)
        
        # Combine all context parts
        context = "\n".join(context_parts)
        
        # If no results found, note this in the prompt
        if not results:
            context = "No relevant documentation was found for this query."
        
        # Create system prompt
        system_prompt = """
        You are a helpful documentation assistant for Kafka, React, and Spark. 
        When answering questions:
        1. Only use information from the provided documentation excerpts if available
        2. Always include citations in [citation:X] format where X is the document number when using information from documents
        3. If documents contradict each other, mention the discrepancy
        4. If the question cannot be answered from the provided documents, say so clearly
        5. Format your response in a clear, concise manner
        6. For 'how-to' questions, structure answers as ordered steps when appropriate
        7. If no documentation is provided or the query is conversational (like greetings), respond in a friendly, helpful way explaining that you're a documentation assistant for Kafka, React, and Spark
        """
        
        # Create user prompt
        user_prompt = f"""
        Based on the following documentation excerpts, please answer the question: "{question}"
        
        {context}
        
        Please provide a comprehensive and accurate answer. When you use information from the documents, 
        include a citation in this format: [citation:X] where X is the document number.
        Focus only on information provided in the documents.
        """
        
        # Call the OpenAI API
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using gpt-3.5-turbo for development
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2  # Lower temperature for more deterministic responses
            )
            
            return response.choices[0].message.content
        except Exception as e:
            # Re-raise the exception for the caller to handle
            raise Exception(f"OpenAI API call failed: {str(e)}")
            
    def _legacy_generate_answer(self, question, results):
        """
        Legacy template-based answer generation method. Not used in the current implementation.
        Kept for reference only.
        
        Args:
            question (str): User question
            results (list): Search results
            
        Returns:
            str: Generated answer
        """
        # This was the original template-based implementation
        # It is no longer used and kept only for reference
        pass
    
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
