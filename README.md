# Documentation AI Assistant

This is a smart generative AI solution that works with documentation from Kafka, React, and Spark to answer questions with proper citations.

## Features

- Processes HTML and Markdown documentation files
- Creates searchable indices for efficient information retrieval
- Answers questions using relevant content from documentation
- Provides proper citations to source documentation
- Offers a web interface for easy interaction

## Requirements

- Python 3.10+
- Flask
- BeautifulSoup4
- Markdown
- FAISS
- scikit-learn
- NLTK

## Installation

1. Clone the repository or extract the provided files
2. Install the required dependencies:

```bash
pip install flask beautifulsoup4 markdown faiss-cpu scikit-learn nltk
```

## Usage

### Processing Documentation

Before using the AI assistant, you need to process the documentation files:

1. Place your documentation files in a directory (HTML and/or Markdown formats)
2. Run the document processing script:

```bash
cd document_processor
python process_documents.py
```

This will:
- Parse all documentation files
- Extract content and metadata
- Create document chunks
- Build a searchable index
- Save the processed documents and index

### Running the Web Interface

To start the web interface:

```bash
cd document_processor
python app.py
```

This will start a Flask server on port 5000. You can access the web interface at:
- http://localhost:5000 (local access)
- Or via the exposed port URL (if using a cloud environment)

### Using the API Directly

You can also use the API directly:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the compiler?"}' http://localhost:5000/api/answer
```

## Project Structure

- `document_parser.py`: Parses HTML and Markdown documents
- `document_indexer.py`: Creates searchable indices from processed documents
- `query_answerer.py`: Answers questions using the document index
- `process_documents.py`: Main script to process all documentation
- `app.py`: Flask web application
- `templates/index.html`: Web interface template
- `static/`: Static files for the web interface
- `processed_files/`: Directory containing processed documents and indices:
  - `extracted_raw_documents.json`: Extracted content from raw documents
  - `search_index/`: Directory containing search index files

## Example Questions

The system can answer questions like:
- "What is the compiler?"
- "How do I get started developing a UI?"
- "How to secure my cluster?"
- "How to update the SSL keystore?"
- "Is streaming supported?"

## Document Processing Pipeline

The document processing system consists of two main components that work in sequence:

1. **Document Parser** (`document_parser.py`):
   - **Purpose**: Extracts and processes raw content from HTML and Markdown files
   - **Input**: Raw document files (.html, .md)
   - **Output**: Structured JSON data with extracted content and metadata
   - **Process**: Parses files → Extracts text and metadata → Saves to `extracted_raw_documents.json` in processed_files directory

2. **Document Indexer** (`document_indexer.py`):
   - **Purpose**: Creates searchable indices from processed documents
   - **Input**: Processed JSON data from the parser (`extracted_raw_documents.json`)
   - **Output**: Vector-based search index for efficient document retrieval
   - **Process**: Loads processed data → Chunks documents → Creates TF-IDF vectors → Builds FAISS index in `search_index` directory

This two-stage pipeline allows for efficient document processing and retrieval, separating the concerns of content extraction and search indexing.

## How It Works

1. **Document Processing**:
   - HTML Parser extracts text from Kafka HTML documentation
   - Markdown Parser extracts text from React and Spark Markdown documentation
   - Document Indexer creates searchable index of all content
   - Metadata Extractor captures document information for citations

2. **Query Processing**:
   - Query Analyzer identifies key terms in user questions
   - Search Engine finds relevant document chunks
   - Context Builder assembles information from search results
   - Response Generator creates comprehensive answers
   - Citation Manager adds proper citations to responses

3. **User Interface**:
   - Web Interface allows users to input questions and view responses
   - API Endpoint provides programmatic access to the system

## Customization

You can customize the system by:
- Adding more documentation files to the base directory
- Adjusting chunk size in `document_indexer.py`
- Modifying search parameters in `query_answerer.py`
- Changing the UI design in `templates/index.html`

## Troubleshooting

- If you encounter NLTK errors, ensure you've downloaded the required data:
  ```python
  import nltk
  nltk.download('punkt')
  ```

- If the web interface doesn't load, check that Flask is running and the port is accessible

- For large documentation sets, increase the memory allocation if needed
