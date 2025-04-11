"""
Document Parser Module

This module provides functionality to parse HTML and Markdown documents,
extract their content, and prepare them for indexing and searching.
"""

import os
import re
import json
from bs4 import BeautifulSoup
import markdown
from pathlib import Path

class DocumentParser:
    """Parser for HTML and Markdown documents."""
    
    def __init__(self, base_dir):
        """
        Initialize the document parser.
        
        Args:
            base_dir (str): Base directory containing the documents
        """
        self.base_dir = base_dir
        self.documents = []
    
    def parse_html(self, file_path):
        """
        Parse HTML document and extract content.
        
        Args:
            file_path (str): Path to the HTML file
            
        Returns:
            dict: Document metadata and content
        """
        rel_path = os.path.relpath(file_path, self.base_dir)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.text if title_tag else os.path.basename(file_path)
        
        # Extract content (remove script and style elements)
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text (remove extra whitespace)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Create document object
        document = {
            'id': len(self.documents),
            'title': title,
            'content': text,
            'source_file': rel_path,
            'format': 'html',
            'technology': self._determine_technology(rel_path)
        }
        
        return document
    
    def parse_markdown(self, file_path):
        """
        Parse Markdown document and extract content.
        
        Args:
            file_path (str): Path to the Markdown file
            
        Returns:
            dict: Document metadata and content
        """
        rel_path = os.path.relpath(file_path, self.base_dir)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Convert markdown to HTML
        html = markdown.markdown(content)
        
        # Parse HTML to extract text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title (first h1 or filename)
        title_tag = soup.find('h1')
        title = title_tag.text if title_tag else os.path.basename(file_path)
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text (remove extra whitespace)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Create document object
        document = {
            'id': len(self.documents),
            'title': title,
            'content': text,
            'source_file': rel_path,
            'format': 'markdown',
            'technology': self._determine_technology(rel_path)
        }
        
        return document
    
    def _determine_technology(self, rel_path):
        """
        Determine the technology based on the file path.
        
        Args:
            rel_path (str): Relative path of the document
            
        Returns:
            str: Technology name (kafka, react, or spark)
        """
        path_parts = rel_path.split(os.sep)
        if path_parts[0] == 'kafka':
            return 'kafka'
        elif path_parts[0] == 'react':
            return 'react'
        elif path_parts[0] == 'spark':
            return 'spark'
        else:
            return 'unknown'
    
    def process_directory(self, directory=None):
        """
        Process all documents in the specified directory.
        
        Args:
            directory (str, optional): Directory to process. Defaults to base_dir.
            
        Returns:
            list: List of processed documents
        """
        if directory is None:
            directory = self.base_dir
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip non-document files
                if not (file.endswith('.html') or file.endswith('.md')):
                    continue
                
                # Parse document based on file extension
                if file.endswith('.html'):
                    document = self.parse_html(file_path)
                    self.documents.append(document)
                elif file.endswith('.md'):
                    document = self.parse_markdown(file_path)
                    self.documents.append(document)
        
        return self.documents
    
    def save_documents(self, output_file):
        """
        Save processed documents to a JSON file.
        
        Args:
            output_file (str): Path to the output JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(self.documents, file, indent=2)
        
        print(f"Saved {len(self.documents)} documents to {output_file}")


if __name__ == "__main__":
    # Example usage
    base_dir = "/home/ubuntu/interview_analysis"
    parser = DocumentParser(base_dir)
    documents = parser.process_directory()
    parser.save_documents("processed_files/extracted_raw_documents.json")
