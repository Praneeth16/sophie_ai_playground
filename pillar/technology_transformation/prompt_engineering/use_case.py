import streamlit as st
import os
import requests
import hashlib
import json

# Set page config
st.set_page_config(page_title="Claude Prompt Generator", layout="wide")

# Load CSS from external file
def load_css():
    """Load CSS from external file"""
    try:
        with open("artifacts/styles.css", "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure artifacts/styles.css exists.")

# Load CSS
load_css()

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'task_description' not in st.session_state:
        st.session_state.task_description = ""
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []

# Cache function for API responses
@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_prompt(task_description, api_key):
    """Cached function to generate prompts from Anthropic API"""
    # Create a hash for the cache key
    cache_key = hashlib.md5(f"{task_description}".encode()).hexdigest()
    
    API_URL = "https://api.anthropic.com/v1/experimental/generate_prompt"
    HEADERS = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": task_description.strip()}]
            }
        ],
        "model": "claude-3-7-sonnet-20250219"
    }
    
    try:
        res = requests.post(API_URL, headers=HEADERS, json=payload)
        res.raise_for_status()
        generated = res.json().get("messages", [{}])[0].get("content", [{}])[0].get("text", "")
        return generated, None
    except requests.exceptions.RequestException as e:
        return None, str(e)

# Initialize session state
init_session_state()

# Header section with hero styling
st.markdown('<div class="hero-title">Claude Prompt Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Generate optimized prompts from your task descriptions using Claude</div>', unsafe_allow_html=True)

# Load API key securely
API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")

# Main content area
st.markdown('<div class="arsenal-title">Prompt Configuration</div>', unsafe_allow_html=True)

# Task description input
st.markdown("**Describe the task you want to create a prompt for**")
task_description = st.text_area(
    "Task Description",
    height=200, 
    placeholder="e.g., I want to create a prompt that helps users write professional emails to clients",
    value=st.session_state.task_description,
    key="task_description_input",
    label_visibility="collapsed"
)
st.session_state.task_description = task_description

# Generate button
if st.button("Generate Prompt", type="primary", use_container_width=True):
    if not API_KEY:
        st.error("Please set your Anthropic API key in your environment variables or in st.secrets.")
    elif not task_description.strip():
        st.warning("Please describe the task you want to create a prompt for.")
    else:
        with st.spinner("Generating your prompt..."):
            generated_prompt, error = generate_prompt(task_description, API_KEY)
            
            if error:
                st.error(f"API error: {error}")
            elif generated_prompt:
                # Store in history
                generation_record = {
                    "task": task_description,
                    "generated_prompt": generated_prompt
                }
                st.session_state.generation_history.append(generation_record)
                
                st.success("Prompt generated successfully!")
            else:
                st.error("No prompt was generated.")

# Results section
if st.session_state.generation_history:
    st.markdown('<div class="arsenal-title">Generated Prompt</div>', unsafe_allow_html=True)
    
    # Show latest result
    latest_result = st.session_state.generation_history[-1]
    
    st.markdown("**Your Generated Prompt:**")
    st.code(latest_result["generated_prompt"], language="markdown")
    
    # Copy button for generated prompt
    if st.button("Copy Generated Prompt", use_container_width=True):
        st.code(latest_result["generated_prompt"], language="markdown")
        st.info("Generated prompt displayed above for easy copying.")

# History section
if len(st.session_state.generation_history) > 1:
    st.markdown('<div class="arsenal-title">Generation History</div>', unsafe_allow_html=True)
    
    # Show history in expandable sections
    for i, record in enumerate(reversed(st.session_state.generation_history[:-1]), 1):
        with st.expander(f"Generation {len(st.session_state.generation_history) - i}"):
            st.markdown("**Task Description:**")
            st.write(record["task"])
            st.markdown("**Generated Prompt:**")
            st.code(record["generated_prompt"], language="markdown")

# Clear history button
if st.session_state.generation_history:
    if st.button("Clear History", help="Clear all generation history"):
        st.session_state.generation_history = []
        st.rerun()
