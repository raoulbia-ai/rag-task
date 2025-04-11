"""
Flask Web Application for Documentation AI Assistant

This module provides a web interface for the AI assistant that answers
questions using documentation with proper citations.
"""

from flask import Flask, request, jsonify, render_template
import os
import sys
import json
from query_answerer import QueryAnswerer

app = Flask(__name__)

# Initialize the query answerer
index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_files", "search_index")
answerer = QueryAnswerer(index_dir)

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
    
    # Get answer from query answerer
    result = answerer.answer_question(question)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
