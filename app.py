"""
Flask Web Application for Documentation AI Assistant

This module provides a web interface for the AI assistant that answers
questions using documentation with proper citations.
"""

from flask import Flask, request, jsonify, render_template
import os
import sys
import json
from dotenv import load_dotenv
from query_answerer import QueryAnswerer

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize the query answerer
index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_files", "search_index")
use_llm = os.environ.get("OPENAI_API_KEY") is not None
answerer = QueryAnswerer(index_dir, use_llm=use_llm)

# Log LLM status
if use_llm:
    print("LLM integration enabled using OpenAI API")
else:
    print("Warning: OpenAI API key not found. LLM integration disabled.")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/answer', methods=['POST'])
def answer():
    """API endpoint to answer questions."""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({
            'error': 'No question provided'
        }), 400
    
    try:
        # Get answer from query answerer
        result = answerer.answer_question(question)
        
        # If there was an error generated inside the answerer
        if 'error' in result:
            return jsonify(result), 500
            
        return jsonify(result)
    except Exception as e:
        # Handle any unexpected exceptions
        error_message = f"Error processing question: {str(e)}"
        print(error_message)
        return jsonify({
            'question': question,
            'answer': "An unexpected error occurred while processing your question.",
            'citations': [],
            'error': error_message
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
