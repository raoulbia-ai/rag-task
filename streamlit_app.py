"""
Streamlit Web Application for Documentation AI Assistant

This module provides a Streamlit chat interface for the RAG assistant that answers
questions using documentation with proper citations.
"""

import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
from query_answerer import QueryAnswerer
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Documentation AI Assistant",
    page_icon="📚",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .citation {
        font-size: 0.85rem;
        color: #555;
        border-left: 3px solid #3498db;
        padding-left: 10px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_query_answerer():
    """Initialize and cache the query answerer to avoid reloading on each rerun"""
    index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_files", "search_index")
    use_llm = os.environ.get("OPENAI_API_KEY") is not None
    
    if use_llm:
        st.sidebar.success("✅ LLM integration enabled using OpenAI API")
    else:
        st.sidebar.error("❌ OpenAI API key not found. LLM integration disabled.")
        
    return QueryAnswerer(index_dir, use_llm=use_llm)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    return None

openai_client = get_openai_client()

# Initialize query answerer
answerer = initialize_query_answerer()

def consider_conversation_context(query, conversation_history=[]):
    """
    Consider conversation context to enhance the query if needed.
    
    Args:
        query (str): The user's current query
        conversation_history (list): Previous messages
        
    Returns:
        str: An enhanced query with conversation context if relevant
    """
    # Skip enhancement if no conversation history or no LLM
    if len(conversation_history) <= 1 or not openai_client:
        return query
    
    # Only consider last few meaningful exchanges
    recent_history = []
    for msg in conversation_history[-6:]:  # Last 6 messages
        if not msg.get("is_clarity_check"):
            recent_history.append({"role": msg["role"], "content": msg["content"]})
    
    # If not enough context, return original query
    if len(recent_history) <= 1:
        return query
    
    # Check for references to previous context
    context_keywords = [
        "it", "that", "this", "these", "those", "they", "them", 
        "previous", "earlier", "above", "mentioned", "said",
        "same", "like", "also", "again", "more"
    ]
    
    # Simple heuristic - check if query contains context keywords 
    # and is relatively short (likely needs context)
    has_context_keywords = any(keyword in query.lower().split() for keyword in context_keywords)
    is_short_query = len(query.split()) < 7
    
    if not (has_context_keywords or is_short_query):
        return query  # No need for enhancement
    
    # Create system prompt for context-aware query enhancement
    system_prompt = """
    You are an AI assistant that enhances user queries with conversation context.
    Given a conversation history and current query, your task is to:
    
    1. Determine if the query refers to or depends on previous conversation
    2. If it does, create an expanded query that includes necessary context
    3. If it doesn't need context, return the original query unchanged
    
    Focus on documentation topics: Kafka, React, and Spark.
    
    RESPOND ONLY with the enhanced query or original query - no explanations or other text.
    """
    
    # Format the conversation history for the prompt
    history_text = ""
    for msg in recent_history[:-1]:  # Exclude the current query
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n\n"
    
    user_prompt = f"""
    Conversation history:
    {history_text}
    
    Current query: {query}
    
    If this query needs context from the conversation to be properly understood, 
    enhance it to include that context. Otherwise, return it unchanged.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        enhanced_query = response.choices[0].message.content.strip()
        
        # Log what happened for debugging
        if enhanced_query != query:
            print(f"Enhanced query: '{query}' → '{enhanced_query}'")
        
        return enhanced_query
    except Exception as e:
        print(f"Error enhancing query with context: {e}")
        return query  # Return original on error

def reformulate_query(original_query, clarification, conversation_history=[]):
    """
    Reformulate a query based on the original question, clarification, and conversation history.
    
    Args:
        original_query (str): The original user query
        clarification (str): The user's response to the follow-up question
        conversation_history (list): Previous messages
        
    Returns:
        str: A reformulated query for better retrieval
    """
    if not openai_client:
        # If no OpenAI client available, just combine the queries
        return f"{original_query} {clarification}"
    
    # Get relevant conversation context (last few messages)
    recent_messages = []
    for msg in conversation_history[-6:]:  # Last 6 messages
        if not msg.get("is_clarity_check"):
            recent_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Create system prompt for query reformulation
    system_prompt = """
    You are an AI assistant that helps reformulate user queries to search documentation more effectively.
    Your task is to create a clear, specific search query based on:
    1. The user's original ambiguous question
    2. Their clarification response
    3. Any relevant context from the conversation history
    
    Focus on these documentation topics: Kafka, React, and Spark.
    
    Guidelines:
    - Create a single, coherent query that combines all relevant information
    - Include specific terms that would likely appear in technical documentation
    - If the user mentions a specific technology (Kafka, React, or Spark), ensure it's prominent in the query
    - If appropriate, include terms like "how to", "configuration", "example", etc.
    - The query should be 1-3 sentences, focused and specific
    
    RESPOND ONLY with the reformulated query - no explanations or other text.
    """
    
    # Provide context and the current exchange
    user_prompt = f"""
    Original ambiguous question: {original_query}
    
    Clarification from user: {clarification}
    
    Conversation context:
    {json.dumps(recent_messages, indent=2)}
    
    Please reformulate this into a clear, specific search query for documentation.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        reformulated_query = response.choices[0].message.content.strip()
        print(f"Reformulated query: {reformulated_query}")  # For debugging
        return reformulated_query
    except Exception as e:
        print(f"Error reformulating query: {e}")
        # Fall back to simple combination
        return f"{original_query} {clarification}"

def check_query_clarity(query, conversation_history=[]):
    """
    Check if a query is clear enough or needs clarification.
    
    Args:
        query (str): The user's query
        conversation_history (list): Previous messages
        
    Returns:
        dict: Contains is_clear (bool) and follow_up_question (str) if needed
    """
    if not openai_client:
        # If no OpenAI client, assume all queries are clear
        return {"is_clear": True}
    
    # Convert conversation history to format expected by OpenAI
    formatted_history = []
    for msg in conversation_history:
        # Skip messages about clarity checks
        if msg.get("is_clarity_check"):
            continue
        formatted_history.append({"role": msg["role"], "content": msg["content"]})
    
    # Create system prompt
    system_prompt = """
    You are an AI assistant helping users with documentation for Kafka, React, and Spark.
    Your task is to determine if a user's question is specific and clear enough to search for an answer in documentation.
    
    If the question is ambiguous, vague, or missing critical details:
    1. Respond with {"is_clear": false, "follow_up_question": "Your specific follow-up question here"}
    2. Your follow-up question should ask for the SINGLE most important missing piece of information
    
    If the question is clear and specific enough to search documentation:
    1. Respond with {"is_clear": true}
    
    Examples of unclear questions that need follow-up:
    - "How do I configure it?" (What is "it"? Need to know which system)
    - "What's the best way?" (Best way for what? Need specific goal/context)
    - "How does this work?" (What is "this"? Need specific feature/concept)
    
    Examples of clear questions:
    - "How do I configure SSL in Kafka?"
    - "What React hooks are available for handling form state?"
    - "What's the syntax for Spark SQL window functions?"
    
    IMPORTANT: ONLY respond with the JSON object, nothing else.
    """
    
    # Add context about the conversation history
    if formatted_history:
        history_context = "Consider this conversation history:\n"
        for msg in formatted_history[-3:]:  # Only include latest 3 messages
            history_context += f"{msg['role']}: {msg['content']}\n"
        user_prompt = f"{history_context}\nNew question: {query}\n\nIs this question clear enough to search for in documentation?"
    else:
        user_prompt = f"Question: {query}\n\nIs this question clear enough to search for in documentation?"
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )
        
        # Extract JSON from response
        result_text = response.choices[0].message.content.strip()
        # Handle potential non-JSON responses
        try:
            # Extract JSON if it's wrapped in code blocks or has extra text
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError:
            # If response isn't valid JSON, default to assuming query is clear
            return {"is_clear": True}
    except Exception as e:
        print(f"Error checking query clarity: {e}")
        # If API call fails, default to assuming query is clear
        return {"is_clear": True}

# App header
st.markdown("<h1 class='main-header'>Documentation AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("Ask questions about **Kafka**, **React**, or **Spark** documentation.")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "awaiting_followup" not in st.session_state:
    st.session_state.awaiting_followup = False
    
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Show example questions as buttons at the beginning if no messages yet
if not st.session_state.messages:
    st.markdown("### Try these example questions:")
    example_cols = st.columns(2)
    example_questions = [
        "What is the compiler?",
        "How do I get started developing a UI?",
        "How to secure my cluster?", 
        "How to update the SSL keystore?",
        "Is streaming supported?"
    ]
    
    for i, example in enumerate(example_questions):
        col = example_cols[i % 2]
        if col.button(example, key=f"example_{i}"):
            st.session_state.messages.append({"role": "user", "content": example})
            # Force a rerun to show the new message
            st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        
        # If this is an assistant message with citations, show them
        if message.get("role") == "assistant" and message.get("citations"):
            st.markdown("**Sources:**")
            for citation in message.get("citations", []):
                st.markdown(
                    f"<div class='citation'>[{citation['id']}] <b>{citation['title']}</b> "
                    f"({citation['technology']}) - {citation['source_file']}</div>",
                    unsafe_allow_html=True
                )

# Chat input
prompt = st.chat_input("Ask a question about Kafka, React, or Spark...")

# Handle the user input
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if we were waiting for a follow-up response
    if st.session_state.awaiting_followup:
        # This is a response to our follow-up question
        original_query = st.session_state.current_query
        
        # Use LLM to reformulate the query based on conversation context
        reformulated_query = reformulate_query(original_query, prompt, st.session_state.messages)
        
        # Reset the follow-up state
        st.session_state.awaiting_followup = False
        st.session_state.current_query = ""
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                # Get answer using the reformulated query
                result = answerer.answer_question(reformulated_query)
                
                # Display answer
                st.markdown(result['answer'], unsafe_allow_html=True)
                
                # Display citations if any
                if result['citations']:
                    st.markdown("**Sources:**")
                    for citation in result['citations']:
                        st.markdown(
                            f"<div class='citation'>[{citation['id']}] <b>{citation['title']}</b> "
                            f"({citation['technology']}) - {citation['source_file']}</div>",
                            unsafe_allow_html=True
                        )
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result['answer'],
            "citations": result['citations']
        })
    else:
        # This is a new query - check if it's clear enough
        clarity_result = check_query_clarity(prompt, st.session_state.messages)
        
        if clarity_result.get("is_clear", True):
            # If the query is clear, proceed with answering
            with st.chat_message("assistant"):
                with st.spinner("Searching documentation..."):
                    # Check if this query might benefit from conversation context
                    enhanced_query = consider_conversation_context(prompt, st.session_state.messages)
                    
                    # Get answer from query answerer with the enhanced query
                    result = answerer.answer_question(enhanced_query)
                    
                    # Display answer
                    st.markdown(result['answer'], unsafe_allow_html=True)
                    
                    # Display citations if any
                    if result['citations']:
                        st.markdown("**Sources:**")
                        for citation in result['citations']:
                            st.markdown(
                                f"<div class='citation'>[{citation['id']}] <b>{citation['title']}</b> "
                                f"({citation['technology']}) - {citation['source_file']}</div>",
                                unsafe_allow_html=True
                            )
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "citations": result['citations']
            })
        else:
            # If the query needs clarification, ask a follow-up question
            follow_up = clarity_result.get("follow_up_question", "Could you please provide more details?")
            
            with st.chat_message("assistant"):
                st.markdown(follow_up)
            
            # Store that we're waiting for a follow-up and the original query
            st.session_state.awaiting_followup = True
            st.session_state.current_query = prompt
            
            # Add the follow-up question to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": follow_up,
                "is_clarity_check": True
            })

# Add sidebar information
with st.sidebar:
    st.markdown("## About")
    st.markdown(
        "This AI assistant uses documentation from Kafka, React, and Spark to answer your questions."
    )
    st.markdown("## Conversation Features")
    st.markdown(
        "This assistant can:\n"
        "- Ask follow-up questions when your query is unclear\n"
        "- Remember context from previous messages\n"
        "- Provide citations to documentation sources\n"
        "- Handle multi-turn conversations"
    )
    
    st.markdown("## How it works")
    st.markdown(
        "1. Your question is analyzed for clarity\n"
        "2. If needed, follow-up questions are asked\n"
        "3. Relevant documentation sections are retrieved\n"
        "4. Context from the conversation is considered\n"
        "5. An AI generates a comprehensive answer\n"
        "6. Citations link back to the original documents"
    )
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.awaiting_followup = False
        st.session_state.current_query = ""
        st.rerun()

if __name__ == "__main__":
    # This is used when running the file directly
    # Streamlit uses a different entry point when run with `streamlit run`
    pass