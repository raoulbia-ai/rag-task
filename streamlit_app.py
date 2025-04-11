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
        # Combine the original query with this clarification
        original_query = st.session_state.current_query
        combined_query = f"Original question: {original_query}\nClarification: {prompt}"
        
        # Reset the follow-up state
        st.session_state.awaiting_followup = False
        st.session_state.current_query = ""
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                # Get answer using the combined query
                result = answerer.answer_question(combined_query)
                
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
                    # Get answer from query answerer
                    result = answerer.answer_question(prompt)
                    
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
    st.markdown("## How it works")
    st.markdown(
        "1. Your question is used to search the documentation\n"
        "2. Relevant sections are retrieved\n"
        "3. An AI generates a comprehensive answer\n"
        "4. Citations link back to the original documents"
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