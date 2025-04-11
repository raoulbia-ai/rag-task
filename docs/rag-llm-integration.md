# LLM Integration for RAG System

This document outlines a plan to integrate a Large Language Model (LLM) into the existing RAG (Retrieval Augmented Generation) system, which currently uses a template-based approach for answer generation.

## Current System Analysis

The current implementation has:
- Document parsing for HTML and Markdown files
- Document chunking with sentence boundaries
- TF-IDF vectors and FAISS index for retrieval
- Template-based answer generation (not using an LLM)
- Citation tracking and presentation
- Web interface with Flask

## LLM Integration Steps

### 1. Choose an LLM Provider/API

Options include:
- **OpenAI (GPT models)**: Most popular, good documentation
- **Anthropic (Claude)**: Strong at following instructions and context handling
- **HuggingFace**: Various open-source models
- **Local open-source models**: Like Llama, Mistral, etc. (requires computing resources)

### 2. Update Dependencies

Add the appropriate client library to `requirements.txt`:

```
# Original dependencies
flask
beautifulsoup4
markdown
faiss-cpu
scikit-learn
nltk

# New dependencies (choose based on selected LLM)
openai
anthropic
transformers
langchain  # Optional but helpful for RAG workflows
python-dotenv  # For environment variable management
```

### 3. Modify the Query Answerer Implementation

Update `query_answerer.py` to use an LLM for answer generation:

```python
def _generate_answer(self, question, results, max_tokens=1024):
    """
    Generate an answer using an LLM based on the question and search results.
    
    Args:
        question (str): User question
        results (list): Search results
        max_tokens (int): Maximum length of generated response
        
    Returns:
        str: Generated answer with citations
    """
    # Skip if no results found
    if not results:
        return "I couldn't find relevant information to answer this question in the provided documentation."
    
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
    
    # Create prompt for the LLM
    prompt = f"""
    Based on the following documentation excerpts, please answer the question: "{question}"
    
    {context}
    
    Please provide a comprehensive and accurate answer. When you use information from the documents, 
    include a citation in this format: [citation:X] where X is the document number.
    Focus only on information provided in the documents.
    """
    
    # Call the LLM API
    response = self._call_llm_api(prompt, max_tokens)
    
    return response
```

### 4. Implement LLM API Call Function

Add a method to call your chosen LLM API:

```python
def _call_llm_api(self, prompt, max_tokens=1024):
    """
    Call the LLM API to generate a response.
    
    Args:
        prompt (str): The prompt to send to the LLM
        max_tokens (int): Maximum response length
    
    Returns:
        str: Generated response
    """
    # Example for OpenAI
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # or whichever model you prefer
            messages=[
                {"role": "system", "content": "You are a documentation assistant that helps users understand Kafka, React, and Spark. Always cite your sources using [citation:X] format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2  # Lower temperature for more deterministic responses
        )
        
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to template approach if API call fails
        print(f"LLM API call failed: {e}")
        return self._fallback_generate_answer(question, results)
```

### 5. Set Up API Key Management

Add secure API key management:

```python
# In app.py or a configuration file
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Check if API key is available
if not os.environ.get("OPENAI_API_KEY"):  # or ANTHROPIC_API_KEY, etc.
    print("Warning: LLM API key not found. Falling back to template-based responses.")
```

### 6. Add a Configuration Option

Allow toggling between LLM and template-based answers:

```python
# In QueryAnswerer class
def __init__(self, index_dir, use_llm=True):
    """
    Initialize the query answerer.
    
    Args:
        index_dir (str): Directory containing the document index
        use_llm (bool): Whether to use LLM for answer generation
    """
    self.index_dir = index_dir
    self.indexer = DocumentIndexer.load_index(index_dir)
    self.use_llm = use_llm and os.environ.get("OPENAI_API_KEY") is not None  # Only use LLM if API key is available
    
    # Load chunks for citation information
    with open(os.path.join(index_dir, 'chunks.json'), 'r', encoding='utf-8') as file:
        self.chunks = json.load(file)
```

### 7. Update the Answer Generation Logic

Modify the `answer_question` method to use the LLM when available:

```python
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
    
    # Generate answer using LLM or template approach
    if self.use_llm:
        answer = self._generate_answer(question, context_results)
    else:
        answer = self._template_generate_answer(question, context_results)  # Rename the original method
    
    # Create citations
    citations = self._create_citations(context_results)
    
    return {
        'question': question,
        'answer': answer,
        'citations': citations
    }
```

### 8. Refine the Prompt Engineering

Experiment with different prompt structures:

```python
# More sophisticated system prompt
system_prompt = """
You are a helpful documentation assistant for Kafka, React, and Spark. 
When answering questions:
1. Only use information from the provided documentation excerpts
2. Always include citations in [citation:X] format where X is the document number
3. If documents contradict each other, mention the discrepancy
4. If the question cannot be answered from the provided documents, say so clearly
5. Format your response in a clear, concise manner
6. For 'how-to' questions, structure answers as ordered steps when appropriate
"""
```

### 9. Add Error Handling and Fallbacks

Ensure robust handling of API failures by renaming the original template-based method:

```python
def _template_generate_answer(self, question, results):
    """
    Generate an answer based on templates (original implementation).
    Used as fallback when LLM is unavailable.
    
    Args:
        question (str): User question
        results (list): Search results
        
    Returns:
        str: Generated answer
    """
    # This is the original _generate_answer method code
    # Extract technology from top result
    top_technology = results[0]['chunk']['technology'] if results else "unknown"
    
    # Create answer introduction based on question type
    if re.search(r'what is|what are|explain|describe', question.lower()):
        intro = f"Based on the {top_technology} documentation, "
    # ... rest of the original implementation
```

### 10. Advanced RAG Pipeline Considerations

For production use, consider:

- Using a chunking strategy that preserves more context
- Switching from TF-IDF to embeddings-based retrieval using models like:
  - `sentence-transformers/all-MiniLM-L6-v2` (fast, lightweight)
  - `sentence-transformers/all-mpnet-base-v2` (more accurate)
- Implementing re-ranking of retrieved documents
- Adding a post-processing step to verify citations
- Implementing caching for common queries

## Implementation Plan

1. **Phase 1: Basic LLM Integration**
   - Add required dependencies
   - Implement basic LLM call function
   - Set up API key management
   - Add fallback to template approach

2. **Phase 2: Improved Context Building**
   - Refine prompt engineering
   - Optimize context format for better responses
   - Add configuration options for different LLMs

3. **Phase 3: Enhanced Retrieval**
   - Replace TF-IDF with embedding-based retrieval
   - Implement chunk re-ranking
   - Optimize chunk size and overlap

4. **Phase 4: Production Optimization**
   - Add caching for frequent queries
   - Implement answer validation
   - Add usage monitoring and cost controls

## Example Files to Modify

1. `requirements.txt` - Add new dependencies
2. `query_answerer.py` - Implement LLM integration
3. `app.py` - Add configuration options
4. `.env` - Create for API key storage (add to .gitignore)
5. `.gitignore` - Update to exclude .env file

This integration will transform the current template-based system into a true RAG implementation, while maintaining backward compatibility and adding fallback mechanisms for reliability.
