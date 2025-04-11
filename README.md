# Documentation AI Assistant with Conversational RAG

An AI-powered system that answers questions about Kafka, React, and Spark documentation using retrieval augmented generation (RAG) with a conversational interface.

## Features

- **Conversational Interface**: Multi-turn conversations with follow-up questions
- **Documentation Search**: Processes HTML and Markdown documentation files
- **Vector-based Retrieval**: Uses FAISS and TF-IDF for efficient information retrieval
- **LLM Integration**: Uses OpenAI for answer generation and conversation management
- **Citation System**: Provides proper citations to source documentation
- **Streamlit UI**: Interactive chat interface with conversation memory

## Requirements

- Python 3.10+
- Streamlit
- OpenAI API key
- FAISS
- BeautifulSoup4
- Markdown
- scikit-learn
- NLTK
- python-dotenv

## Installation and Setup

### Environment Setup

1. Extract the zip file:
```bash
unzip documentation-assistant.zip
cd documentation-assistant
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n doc-assistant python=3.10
conda activate doc-assistant
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Processing Documentation

Before using the AI assistant, you need to process the documentation files:

1. Place your documentation files in the `files/` directory (HTML and/or Markdown formats)
2. Run the document processing script:

```bash
python process_documents.py
```

This will:
- Parse all documentation files
- Extract content and metadata
- Create document chunks
- Build a searchable index
- Save the processed documents and index

## Running the Application

To start the Streamlit chat interface:

```bash
python -m streamlit run streamlit_app.py
```

This will open a browser window with the chat interface where you can ask questions about the documentation.

## Architecture

### System Components

1. **Document Processing Pipeline**:
   - `document_parser.py`: Parses HTML and Markdown documents
   - `document_indexer.py`: Creates searchable indices from processed documents
   - `process_documents.py`: Main script to process all documentation

2. **Query Processing & LLM Integration**:
   - `query_answerer.py`: Core RAG implementation with LLM integration
   - OpenAI API integration for answer generation

3. **User Interface**:
   - `streamlit_app.py`: Streamlit-based conversational interface

### Conversation Flow

1. **Query Clarity Check**:
   - Incoming user query is analyzed for clarity and specificity
   - If unclear, follow-up questions are generated

2. **Context-Aware Retrieval**:
   - Conversation history is used to enhance retrieval
   - Queries are reformulated based on context when appropriate

3. **RAG Processing**:
   - Relevant documents are retrieved from the vector store
   - Retrieved documents and query are sent to the LLM
   - LLM generates a comprehensive answer with citations

## Prompt Engineering

The system uses several carefully designed prompts:

1. **Query Clarity Prompt**:
   - Determines if a user question is specific enough to search for
   - Generates follow-up questions for ambiguous queries
   - Focuses on extracting the single most important missing piece of information

2. **Query Reformulation Prompt**:
   - Combines original questions with follow-up clarifications
   - Creates coherent search queries that focus on documentation terms
   - Ensures specific technologies (Kafka, React, Spark) are properly prioritized

3. **Context Consideration Prompt**:
   - Detects when a query references previous conversation
   - Enhances queries with relevant context from conversation history
   - Handles pronouns and implicit references

4. **Answer Generation Prompt**:
   - Instructions to use HTML formatting for better display
   - Citation guidelines to only cite when information is used
   - Specific formatting for different response types (lists, code, etc.)

## Known Limitations

1. **Query Reformulation Accuracy**:
   - GPT-3.5-turbo sometimes over-interprets minimal context
   - May connect related technologies even when not explicitly mentioned
   - Example: When asking about "streaming", might pull in both Kafka and Spark contexts

2. **Context Window Limitations**:
   - Limited conversation history due to model context constraints
   - Very long conversations may lose earlier context

3. **Citation Precision**:
   - Citations are based on retrieved documents, not the model's knowledge
   - The system may not always cite the most relevant source

4. **Follow-up Question Quality**:
   - Follow-up questions might sometimes be too generic
   - Multiple rounds may be needed for very ambiguous queries

These limitations are acceptable for an MVP and could be addressed in future versions with more advanced models (GPT-4) or fine-tuning.

## Proposed Improvements

Based on analysis of conversation patterns, the following improvements are recommended for future iterations:

### 1. Enhanced Query Interpretation
- Implement a pre-processing step that analyzes queries for partial technology matches
- Create a mapping of common terms to technologies (e.g., "cluster" → check all technologies)
- Use a lightweight classifier to detect the likely technology domain even with ambiguous terms
- Add common synonyms and related concepts to each technology (e.g., "node", "instance", "cluster")

### 2. Better Handling of Negative Results
- Modify the query clarity prompt to detect domain-specific but vague queries
- When no results are found, have the LLM generate follow-up clarification options first
- Use a confidence threshold - if below a certain level, ask for clarification before responding negatively
- Include "did you mean..." suggestions based on similar terms in the documentation

### 3. More Natural Citations
- Update the HTML formatting in the prompt to use superscript tags for citations
- Create a more elegant styling for citations with smaller, less intrusive formatting
- Consider moving detailed citations to footnotes with hover effects for details
- Use a more natural in-text citation style (e.g., "According to Spark documentation...")

These enhancements could be implemented with relatively minor changes to the existing prompts and UI elements, without requiring architectural changes to the overall system.

## Project Structure

- `document_parser.py`: Parses HTML and Markdown documents
- `document_indexer.py`: Creates searchable indices from processed documents
- `query_answerer.py`: Answers questions using the document index and LLM
- `process_documents.py`: Main script to process all documentation
- `streamlit_app.py`: Streamlit chat interface
- `files/`: Source documentation files
- `processed_files/`: Directory containing processed documents and indices

## Future Enhancements

- Upgrade to embedding-based vector search
- Add re-ranking of retrieved documents
- Implement caching for frequent queries
- Switch to a more advanced LLM like GPT-4
- Add document-grounded fact checking

## Troubleshooting

- If OpenAI API calls fail, verify your API key in the .env file
- For processing errors, ensure NLTK has the required data:
  ```python
  import nltk
  nltk.download('punkt')
  ```
