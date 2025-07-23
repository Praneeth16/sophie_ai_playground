import streamlit as st
import os
import re
import zipfile
import io
import yaml
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
from openai import OpenAI

# Load configuration
with open("pillar/technology_transformation/tool_development/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configure the page
st.set_page_config(
    page_title="Advanced Code Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS from external file
def load_css():
    """Load CSS from external file"""
    try:
        with open("artifacts/styles.css", "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found. Please ensure artifacts/styles.css exists.")

    # Additional custom CSS for sidebar buttons and orange button styling
    custom_css = """
    <style>
    /* Simple and focused sidebar button styling */
    .stSidebar .stButton > button {
        background-color: white !important;
        color: #333333 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    .stSidebar .stButton > button:hover {
        background-color: #f8f9fa !important;
        border-color: #dee2e6 !important;
    }
    
    /* Alternative selector for newer Streamlit versions */
    div[data-testid="stSidebar"] button {
        background-color: white !important;
        color: #333333 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Orange button styling with white/grey text */
    button[kind="primary"] {
        background-color: #ea580c !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #dc2626 !important;
        color: white !important;
    }
    
    /* Form submit button styling */
    .stForm button[kind="primary"] {
        background-color: #ea580c !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .stForm button[kind="primary"]:hover {
        background-color: #dc2626 !important;
        color: white !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Load CSS
load_css()

# API Key from Streamlit Secrets
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    st.error("OPENROUTER_API_KEY not found in Streamlit secrets. Please configure it in .streamlit/secrets.toml")
    st.stop()

# Client Setup
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "chat_history": [],
        "generated_files": [],
        "selected_file": None,
        "project_breakdown": None,
        "generation_approach": "simple",
        "current_step": 1,
        "total_steps": 1,
        "language_choice": "Python",
        "task_breakdown": [],
        "debug_info": []  # Add debug info storage
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def add_debug_info(message: str):
    """Add debug information to session state"""
    st.session_state.debug_info.append(f"{datetime.now().strftime('%H:%M:%S')}: {message}")
    # Keep only last 10 debug messages
    if len(st.session_state.debug_info) > 10:
        st.session_state.debug_info = st.session_state.debug_info[-10:]

def analyze_query_complexity(query: str, language: str) -> Dict[str, Any]:
    """Analyze if the query is simple or complex and break it down"""
    
    analysis_prompt = f"""
    Analyze this coding request and determine if it's simple or complex:
    
    Request: "{query}"
    Target Language: {language}
    
    Respond with a JSON object:
    {{
        "complexity": "simple" or "complex",
        "reasoning": "Brief explanation of why it's simple or complex",
        "breakdown": [
            "Step 1: Description",
            "Step 2: Description",
            ...
        ],
        "estimated_files": 1,
        "technologies": ["{language}", "additional", "technologies"],
        "approach": "direct" or "agent-based",
        "folder_structure": {{
            "simple": ["main file only"],
            "complex": ["src/", "tests/", "docs/", "config/", "requirements.txt"]
        }}
    }}
    
    Simple requests: single file, basic script, simple utility, straightforward example, hello world, basic calculations, simple demonstrations
    Complex requests: full applications, multiple files, database integration, authentication, deployment, APIs, web apps, large systems
    
    For simple requests: 
    - Suggest ONLY the main file (e.g., "hello.py", "calculator.js", "main.go")
    - estimated_files should be 1
    - NO deployment files, NO .gitignore, NO README, NO requirements.txt unless specifically requested
    
    For complex requests: 
    - Suggest proper project structure with organized folders
    - estimated_files should be 3+
    """
    
    try:
        add_debug_info(f"Analyzing query complexity with model: moonshotai/kimi-k2")
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.2
        )
        
        raw_response = response.choices[0].message.content
        add_debug_info(f"Raw analysis response length: {len(raw_response)} chars")
        
        # Try to extract JSON from response (handle cases where it might be wrapped)
        json_str = raw_response
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0]
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0]
        
        analysis = json.loads(json_str)
        add_debug_info(f"Successfully parsed analysis: {analysis['complexity']} complexity")
        return analysis
        
    except json.JSONDecodeError as e:
        add_debug_info(f"JSON parsing failed: {str(e)}")
        st.error(f"Failed to parse analysis response: {str(e)}")
        # Fallback analysis
        return {
            "complexity": "simple",
            "reasoning": "Analysis failed, defaulting to simple approach",
            "breakdown": ["Generate the requested code"],
            "estimated_files": 1,
            "technologies": [language],
            "approach": "direct",
            "folder_structure": {
                "simple": ["main file only"],
                "complex": []
            }
        }
    except Exception as e:
        add_debug_info(f"Analysis error: {str(e)}")
        st.error(f"Error during analysis: {str(e)}")
        # Fallback analysis
        return {
            "complexity": "simple",
            "reasoning": f"Analysis failed due to error: {str(e)}",
            "breakdown": ["Generate the requested code"],
            "estimated_files": 1,
            "technologies": [language],
            "approach": "direct",
            "folder_structure": {
                "simple": ["main file only"],
                "complex": []
            }
        }

def validate_response_content(text: str) -> Dict[str, Any]:
    """Validate if the response contains actual code or just references to attached files"""
    
    # Patterns that indicate the model is not providing direct code
    problematic_patterns = [
        r"(?i)code\s+is\s+attached",
        r"(?i)please\s+find\s+the\s+code\s+attached",
        r"(?i)attached\s+files?",
        r"(?i)see\s+attached",
        r"(?i)code\s+attached",
        r"(?i)files?\s+attached",
        r"(?i)i\s+have\s+attached",
        r"(?i)i\'ve\s+attached",
        r"(?i)uploaded\s+files?",
        r"(?i)download\s+the\s+files?",
        r"(?i)check\s+the\s+attached",
        r"(?i)refer\s+to\s+the\s+attached",
        r"(?i)the\s+files?\s+are\s+attached"
    ]
    
    # Check if response contains problematic patterns
    contains_attachment_reference = any(re.search(pattern, text) for pattern in problematic_patterns)
    
    # Check if response is too short to contain meaningful code
    is_too_short = len(text.strip()) < 50
    
    # Check if response contains typical code patterns
    code_patterns = [
        r"def\s+\w+\(",  # Python functions
        r"function\s+\w+\(",  # JavaScript functions
        r"class\s+\w+",  # Class definitions
        r"import\s+\w+",  # Import statements
        r"package\s+\w+",  # Go/Java packages
        r"#include\s*<",  # C/C++ includes
        r"using\s+\w+",  # C# using statements
        r"public\s+class",  # Java public classes
        r"fn\s+\w+\(",  # Rust functions
        r"<?php",  # PHP opening tag
        r"require\s*[(\'\"]",  # Ruby/Node requires
    ]
    
    contains_code_patterns = any(re.search(pattern, text) for pattern in code_patterns)
    
    # Check for file markers
    has_file_markers = bool(re.search(r"#\s*FILE\s*:", text, re.IGNORECASE))
    
    return {
        "is_valid": not contains_attachment_reference and not is_too_short and (contains_code_patterns or has_file_markers),
        "contains_attachment_reference": contains_attachment_reference,
        "is_too_short": is_too_short,
        "contains_code_patterns": contains_code_patterns,
        "has_file_markers": has_file_markers,
        "content_length": len(text.strip())
    }

def create_explicit_code_prompt(query: str, language: str, attempt_number: int = 1) -> str:
    """Create an explicit prompt that forces direct code output"""
    
    base_prompt = f"""
You are a coding assistant. You MUST provide the actual code directly in your response.

Request: {query}
Language: {language}
Version: {get_language_version(language)}

CRITICAL INSTRUCTIONS:
- DO NOT say "code is attached" or "please find attached" or similar phrases
- DO NOT reference external files or attachments  
- PROVIDE THE ACTUAL CODE DIRECTLY in your response
- Start each file with exactly '# FILE: filename.ext' on its own line
- Include the complete, runnable code after each file marker
- Make it immediately usable without any additional steps

Example format:
# FILE: main.{get_file_extension(language)}
[ACTUAL CODE CONTENT HERE]

# FILE: requirements.txt  
[ACTUAL REQUIREMENTS HERE]

Generate the working code now:
"""
    
    if attempt_number > 1:
        base_prompt += f"""

RETRY ATTEMPT #{attempt_number}:
The previous response was invalid. You MUST include the actual code content directly.
DO NOT mention attachments, files being attached, or external references.
PROVIDE COMPLETE, WORKING CODE in the response body.
"""
    
    return base_prompt

def get_file_extension(language: str) -> str:
    """Get the appropriate file extension for a language"""
    ext_map = {
        "Python": "py",
        "JavaScript": "js", 
        "TypeScript": "ts",
        "Java": "java",
        "Go": "go",
        "Rust": "rs",
        "C#": "cs",
        "PHP": "php",
        "Ruby": "rb",
        "C": "c",
        "C++": "cpp"
    }
    return ext_map.get(language, "txt")

def generate_code_with_retry(query: str, language: str, is_complex: bool = False, max_retries: int = 3) -> List[Tuple[str, str]]:
    """Generate code with retry logic for handling attachment responses"""
    
    for attempt in range(1, max_retries + 1):
        add_debug_info(f"Code generation attempt {attempt}/{max_retries}")
        
        try:
            if is_complex and attempt == 1:
                # Use the complex prompt for first attempt if it's complex
                prompt = config["prompt_template"].format(
                    idea=query,
                    language=language,
                    version=get_language_version(language),
                    extra=f"This is a complex {language} project. PROVIDE ACTUAL CODE DIRECTLY in your response. DO NOT reference attachments."
                )
            else:
                # Use explicit prompt for simple requests or retry attempts
                prompt = create_explicit_code_prompt(query, language, attempt)
            
            add_debug_info(f"Using model: mistralai/devstral-medium for attempt {attempt}")
            response = client.chat.completions.create(
                model="mistralai/devstral-medium",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            generated_code = response.choices[0].message.content
            add_debug_info(f"Generated response length: {len(generated_code)} chars")
            
            # Validate the response
            validation = validate_response_content(generated_code)
            add_debug_info(f"Validation result: {validation}")
            
            if not validation["is_valid"]:
                add_debug_info(f"Invalid response detected on attempt {attempt}")
                
                if validation["contains_attachment_reference"]:
                    add_debug_info("Response contains attachment references")
                    if attempt < max_retries:
                        st.warning(f"⚠️ Attempt {attempt}: AI mentioned 'attached files' instead of providing code. Retrying with more explicit instructions...")
                        continue
                elif validation["is_too_short"]:
                    add_debug_info("Response is too short")
                    if attempt < max_retries:
                        st.warning(f"⚠️ Attempt {attempt}: Response too short ({validation['content_length']} chars). Retrying...")
                        continue
                elif not validation["contains_code_patterns"] and not validation["has_file_markers"]:
                    add_debug_info("Response lacks code patterns")
                    if attempt < max_retries:
                        st.warning(f"⚠️ Attempt {attempt}: No recognizable code patterns found. Retrying...")
                        continue
            
            # Try to parse files
            files = parse_generated_files(generated_code)
            add_debug_info(f"Parsed {len(files)} files from attempt {attempt}")
            
            # Additional validation: check if parsed files contain actual code
            if files:
                valid_files = []
                for filename, content in files:
                    file_validation = validate_response_content(content)
                    if file_validation["is_valid"] or len(content.strip()) > 20:  # More lenient for individual files
                        valid_files.append((filename, content))
                    else:
                        add_debug_info(f"File {filename} failed validation: {file_validation}")
                
                if valid_files:
                    add_debug_info(f"Successfully generated {len(valid_files)} valid files")
                    return valid_files
                else:
                    add_debug_info("No valid files after parsing")
                    if attempt < max_retries:
                        st.warning(f"Attempt {attempt}: Parsed files but none contain valid code. Retrying...")
                        continue
            else:
                add_debug_info("No files could be parsed")
                if attempt < max_retries:
                    st.warning(f"Attempt {attempt}: Could not parse any files from response. Retrying...")
                    continue
            
            # If we reach here and it's the last attempt, show the raw response for debugging
            if attempt == max_retries:
                st.error("All attempts failed. Showing raw response for debugging:")
                with st.expander("Raw AI Response", expanded=True):
                    st.code(generated_code, language="text")
                    st.write("**Validation Details:**")
                    st.json(validation)
                break
                
        except Exception as e:
            add_debug_info(f"Error in attempt {attempt}: {str(e)}")
            if attempt < max_retries:
                st.warning(f"⚠️ Attempt {attempt} failed with error: {str(e)}. Retrying...")
                continue
            else:
                st.error(f"Final attempt failed: {str(e)}")
                break
    
    return []

def generate_simple_code(query: str, language: str) -> List[Tuple[str, str]]:
    """Generate code directly for simple requests using retry logic"""
    return generate_code_with_retry(query, language, is_complex=False)

def generate_complex_code(query: str, language: str, breakdown: List[str]) -> List[Tuple[str, str]]:
    """Generate code using enhanced approach for complex requests with retry logic"""
    return generate_code_with_retry(query, language, is_complex=True)

def get_language_version(language: str) -> str:
    """Get appropriate version string for the language"""
    version_map = {
        "Python": "Python 3.11",
        "JavaScript": "Node.js 18+",
        "TypeScript": "TypeScript 5.0+",
        "Java": "Java 17+",
        "Go": "Go 1.21+",
        "Rust": "Rust 1.70+",
        "C#": ".NET 7+",
        "PHP": "PHP 8.2+",
        "Ruby": "Ruby 3.2+"
    }
    return version_map.get(language, f"{language} latest")

def parse_generated_files(text: str) -> List[Tuple[str, str]]:
    """Parse generated files from model response with enhanced pattern matching"""
    
    add_debug_info(f"Parsing files from text of length: {len(text)}")
    
    # Enhanced patterns to catch more variations
    patterns = [
        # Original patterns with case insensitive matching
        re.compile(r"#\s*FILE\s*:\s*(.*?)\n(.*?)(?=(#\s*FILE\s*:|\Z))", re.DOTALL | re.IGNORECASE),  
        re.compile(r"(?:^|\n)#\s*FILE\s*:\s*(.*?)\n(.*?)(?=(?:\n#\s*FILE\s*:|\Z))", re.DOTALL | re.IGNORECASE),  
        
        # Code block variations
        re.compile(r"```(?:python|javascript|java|go|rust|php|ruby|typescript|csharp|cpp|c)?\n#\s*FILE\s*:\s*(.*?)\n(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```\n#\s*FILE\s*:\s*(.*?)\n(.*?)```", re.DOTALL | re.IGNORECASE),
        
        # Alternative file markers
        re.compile(r"##\s*(.*?)\n```(?:python|javascript|java|go|rust|php|ruby|typescript|csharp|cpp|c)?\n(.*?)```", re.DOTALL),
        re.compile(r"###\s*(.*?)\n```(?:python|javascript|java|go|rust|php|ruby|typescript|csharp|cpp|c)?\n(.*?)```", re.DOTALL),
        re.compile(r"File\s*:\s*(.*?)\n```(?:python|javascript|java|go|rust|php|ruby|typescript|csharp|cpp|c)?\n(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"`(.*?)`\n```(?:python|javascript|java|go|rust|php|ruby|typescript|csharp|cpp|c)?\n(.*?)```", re.DOTALL),
        
        # Filename only patterns (for cases where filename is clearly marked)
        re.compile(r"(?:^|\n)\*\*([^*]+\.(?:py|js|ts|java|go|rs|cs|php|rb|html|css|json|yaml|yml|txt|md))\*\*\n(.*?)(?=(?:\n\*\*[^*]+\.(?:py|js|ts|java|go|rs|cs|php|rb|html|css|json|yaml|yml|txt|md)\*\*|\Z))", re.DOTALL | re.IGNORECASE),
        re.compile(r"(?:^|\n)([^:\n]+\.(?:py|js|ts|java|go|rs|cs|php|rb|html|css|json|yaml|yml|txt|md)):\n(.*?)(?=(?:\n[^:\n]+\.(?:py|js|ts|java|go|rs|cs|php|rb|html|css|json|yaml|yml|txt|md):|\Z))", re.DOTALL | re.IGNORECASE),
    ]
    
    all_matches = []
    
    for i, pattern in enumerate(patterns):
        matches = pattern.findall(text)
        if matches:
            add_debug_info(f"Pattern {i+1} found {len(matches)} matches")
        for match in matches:
            filename = match[0].strip()
            content = match[1].strip()
            if filename and content and len(content) > 10:  # Must have meaningful content
                all_matches.append((filename, content))
    
    # Remove duplicates (same filename) and clean filenames
    seen_files = set()
    files = []
    for filename, content in all_matches:
        clean_filename = filename.strip().replace('`', '').replace('#', '').replace('*', '').strip()
        # Remove any leading/trailing quotes or brackets
        clean_filename = re.sub(r'^["\'\[\(]+|["\'\]\)]+$', '', clean_filename)
        
        if clean_filename not in seen_files and clean_filename and content and not is_attachment_reference(content):
            files.append((clean_filename, content))
            seen_files.add(clean_filename)
    
    add_debug_info(f"Final parsed files: {len(files)}")
    for filename, _ in files:
        add_debug_info(f"  - {filename}")
    
    # If no files parsed but we have content, try a more aggressive approach
    if not files and len(text.strip()) > 100 and not is_attachment_reference(text):
        add_debug_info("No files parsed with patterns, trying fallback approach")
        # Check if the text looks like actual code
        validation = validate_response_content(text)
        if validation["contains_code_patterns"] or validation["has_file_markers"]:
            language_ext = {
                "Python": "py", "JavaScript": "js", "TypeScript": "ts",
                "Java": "java", "Go": "go", "Rust": "rs", "C#": "cs",
                "PHP": "php", "Ruby": "rb", "C": "c", "C++": "cpp"
            }
            ext = language_ext.get(st.session_state.language_choice, "txt")
            fallback_filename = f"main.{ext}"
            files.append((fallback_filename, text.strip()))
            add_debug_info(f"Created fallback file: {fallback_filename}")
    
    return files

def is_attachment_reference(text: str) -> bool:
    """Check if text contains references to attachments instead of actual code"""
    attachment_patterns = [
        r"(?i)code\s+is\s+attached",
        r"(?i)please\s+find\s+attached",
        r"(?i)attached\s+files?",
        r"(?i)see\s+attached",
        r"(?i)download\s+the\s+files?",
        r"(?i)i\s+have\s+attached",
        r"(?i)files?\s+are\s+attached"
    ]
    return any(re.search(pattern, text) for pattern in attachment_patterns)

def render_sidebar_directory_structure(files: List[Tuple[str, str]]):
    """Render directory structure in sidebar with clean expanders and buttons"""
    
    st.sidebar.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Project Structure</div>', unsafe_allow_html=True)
    
    if not files:
        st.sidebar.info("No files generated yet.")
        return
    
    # Create a tree structure
    tree = {}
    for i, (filename, _) in enumerate(files):
        parts = filename.split('/')
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {'type': 'folder', 'children': {}}
            current = current[part]['children']
        current[parts[-1]] = {'type': 'file', 'index': i}
    
    def render_tree(tree_dict):
        for name, value in tree_dict.items():
            if value['type'] == 'folder':
                # Clean folder expander
                with st.sidebar.expander(f"{name}", expanded=True):
                    # Render children inside the expander
                    render_tree(value['children'])
            else:
                # File button - simple and reliable approach
                if st.sidebar.button(f"{name}", key=f"sidebar_file_{value['index']}", use_container_width=True):
                    st.session_state.selected_file = value['index']
                    st.rerun()
    
    # Handle root level files and folders
    root_files = []
    root_folders = {}
    
    for name, value in tree.items():
        if value['type'] == 'file':
            root_files.append((name, value))
        else:
            root_folders[name] = value
    
    # Render root level files first
    for name, value in root_files:
        if st.sidebar.button(f"{name}", key=f"sidebar_file_{value['index']}", use_container_width=True):
            st.session_state.selected_file = value['index']
            st.rerun()
    
    # Render folders with their contents
    for name, value in root_folders.items():
        with st.sidebar.expander(f"{name}", expanded=True):
            render_tree(value['children'])
    
    # Download section in sidebar
    if files:
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Downloads</div>', unsafe_allow_html=True)
        
        if st.session_state.selected_file is not None:
            selected_filename, selected_content = files[st.session_state.selected_file]
            st.sidebar.download_button(
                f"Download {selected_filename.split('/')[-1]}",
                selected_content,
                file_name=selected_filename.split('/')[-1],
                mime="text/plain",
                use_container_width=True
            )
        
        # ZIP download
        zip_data = create_zip_file(files)
        st.sidebar.download_button(
            "Download All Files",
            data=zip_data,
            file_name=f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )

def render_language_selector():
    """Render language selector in sidebar"""
    st.sidebar.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Configuration</div>', unsafe_allow_html=True)
    
    languages = [
        "Python", "JavaScript", "TypeScript", "Java", "Go", 
        "Rust", "C#", "PHP", "Ruby", "Other"
    ]
    
    selected_language = st.sidebar.selectbox(
        "Programming Language:",
        languages,
        index=languages.index(st.session_state.language_choice) if st.session_state.language_choice in languages else 0
    )
    
    if selected_language == "Other":
        custom_language = st.sidebar.text_input("Specify language:", value="")
        if custom_language:
            st.session_state.language_choice = custom_language
    else:
        st.session_state.language_choice = selected_language
    
    # Debug info toggle
    if st.sidebar.checkbox("Show Debug Info", value=False):
        if st.session_state.debug_info:
            st.sidebar.markdown('<div style="font-size: 1rem; font-weight: 600; color: #374151; margin: 0.5rem 0;">Debug Log</div>', unsafe_allow_html=True)
            for debug_msg in st.session_state.debug_info[-5:]:  # Show last 5 messages
                st.sidebar.caption(debug_msg)
    
    return st.session_state.language_choice

def render_task_breakdown_in_chat(breakdown: Dict[str, Any]):
    """Render detailed task breakdown in chat interface"""
    
    # Create a collapsible task breakdown
    with st.expander("📋 Task Breakdown Analysis", expanded=False):
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Complexity", breakdown["complexity"].title())
        with col2:
            st.metric("Approach", breakdown["approach"].replace("-", " ").title())
        with col3:
            st.metric("Est. Files", breakdown["estimated_files"])
        with col4:
            st.metric("Language", st.session_state.language_choice)
        
        # Technologies
        st.markdown('<div style="font-weight: 600; color: #374151; margin: 1rem 0 0.5rem 0;">Technologies & Tools:</div>', unsafe_allow_html=True)
        tech_cols = st.columns(len(breakdown["technologies"]))
        for i, tech in enumerate(breakdown["technologies"]):
            tech_cols[i].markdown(f"`{tech}`")
        
        # Implementation steps
        st.markdown('<div style="font-weight: 600; color: #374151; margin: 1rem 0 0.5rem 0;">Implementation Steps:</div>', unsafe_allow_html=True)
        for i, step in enumerate(breakdown["breakdown"], 1):
            with st.expander(f"Step {i}: {step.split(':')[0] if ':' in step else step}", expanded=False):
                st.write(step)
        
        # Reasoning
        with st.expander("Analysis Reasoning", expanded=False):
            st.write(breakdown["reasoning"])
        
        # Folder structure preview
        structure_key = breakdown["complexity"]
        if structure_key in breakdown.get("folder_structure", {}):
            st.markdown('<div style="font-weight: 600; color: #374151; margin: 1rem 0 0.5rem 0;">Expected Folder Structure:</div>', unsafe_allow_html=True)
            for item in breakdown["folder_structure"][structure_key]:
                st.write(f"• {item}")

def render_code_viewer(files: List[Tuple[str, str]]):
    """Render code viewer in main area"""
    
    if not files:
        st.info("Generate your first project to see the code here!")
        return
        
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #374151; margin-bottom: 1rem; font-family: var(--header-font);">Code Viewer</div>', unsafe_allow_html=True)
    
    if st.session_state.selected_file is not None and st.session_state.selected_file < len(files):
        filename, content = files[st.session_state.selected_file]
        
        # File header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{filename}**")
        with col2:
            file_size = len(content.encode('utf-8'))
            st.caption(f"Size: {file_size} bytes")
        
        # Determine language for syntax highlighting
        ext = filename.split(".")[-1].lower()
        lang_map = {
            "py": "python", "js": "javascript", "ts": "typescript",
            "java": "java", "go": "go", "rs": "rust", "cs": "csharp",
            "php": "php", "rb": "ruby", "sql": "sql", "sh": "shell", 
            "bash": "bash", "yaml": "yaml", "yml": "yaml", "json": "json",
            "html": "html", "css": "css", "md": "markdown"
        }
        language = lang_map.get(ext, "text")
        
        st.code(content, language=language, line_numbers=True)
        
    else:
        st.info("Select a file from the sidebar to view its contents.")

def render_chat_interface():
    """Render chat interface in main area"""
    
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; color: #374151; margin-bottom: 1rem; font-family: var(--header-font);">AI Assistant</div>', unsafe_allow_html=True)
    
    # Info about attachment handling
    with st.expander("ℹ️ How We Handle AI 'Attachment' Responses", expanded=False):
        st.markdown("""
        **Automatic Retry System for Better Code Generation:**
        
        Sometimes AI models respond with phrases like "code is attached" instead of providing actual code. 
        Our system automatically handles this by:
        
        - 🔍 **Detecting** when responses mention "attachments" instead of code
        - ⚡ **Retrying** with more explicit instructions (up to 3 attempts)
        - ✅ **Validating** that responses contain actual code patterns
        - 🔧 **Providing** clear feedback during the retry process
        
        **What You'll See:**
        - ⚠️ Warning messages when retries are happening
        - 📊 Debug information (enable in sidebar) showing validation details
        - ❌ Raw response display if all attempts fail for debugging
        
        This ensures you always get working code files instead of references to non-existent attachments!
        """)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #374151; margin: 1rem 0 0.5rem 0;">Conversation</div>', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    
    # Input form
    st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #374151; margin: 1rem 0 0.5rem 0;">Send Message</div>', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            f"Describe your {st.session_state.language_choice} project:",
            placeholder=f"e.g., Create a {st.session_state.language_choice} web application for task management with user authentication and database integration",
            height=100
        )
        
        submitted = st.form_submit_button("Generate Project", use_container_width=True)
    
    return submitted, user_input

def create_zip_file(files: List[Tuple[str, str]]) -> io.BytesIO:
    """Create a zip file from the generated files"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files:
            zip_file.writestr(filename, content)
    zip_buffer.seek(0)
    return zip_buffer

# Main application layout
def main():
    # Page title following standard format from app.py
    st.markdown('<div class="arsenal-title">Advanced Code Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="arsenal-subtitle">AI-powered development environment with intelligent project analysis</div>', unsafe_allow_html=True)
    st.write("")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar: Language selection and directory structure
    selected_language = render_language_selector()
    render_sidebar_directory_structure(st.session_state.generated_files)
    
    # Main area: Two columns for code viewer and chat
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_code_viewer(st.session_state.generated_files)
    
    with col2:
        submitted, user_input = render_chat_interface()
        
        # Process user input
        if submitted and user_input.strip():
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_input
            })
            
            # Clear debug info for new request
            st.session_state.debug_info = []
            add_debug_info(f"Starting new request: {user_input[:50]}...")
            
            try:
                with st.spinner("Analyzing your request..."):
                    # Step 1: Analyze query complexity
                    breakdown = analyze_query_complexity(user_input, selected_language)
                    st.session_state.project_breakdown = breakdown
                    st.session_state.task_breakdown = breakdown["breakdown"]
                
                # Show task breakdown in chat
                with col2:
                    render_task_breakdown_in_chat(breakdown)
                
                with st.spinner("Generating your project..."):
                    # Step 2: Generate code based on complexity
                    if breakdown["approach"] == "direct":
                        files = generate_simple_code(user_input, selected_language)
                    else:
                        files = generate_complex_code(user_input, selected_language, breakdown["breakdown"])
                    
                    # Update session state
                    st.session_state.generated_files = files
                    st.session_state.selected_file = 0 if files else None
                    
                    # Add completion message to chat
                    if files:
                        completion_message = f"Successfully generated {len(files)} files for your {selected_language} project. Explore the files in the sidebar and view them in the code viewer."
                        add_debug_info(f"Generation completed successfully with {len(files)} files")
                    else:
                        completion_message = f"I generated a response but couldn't parse any files from it. This might be a formatting issue. Please check the debug info or try rephrasing your request."
                        add_debug_info("Generation completed but no files were parsed")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": completion_message
                    })
                    
                    # Force UI refresh
                    st.rerun()
                    
            except Exception as e:
                error_msg = f"Error generating project: {str(e)}"
                add_debug_info(error_msg)
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"I encountered an error while generating your project: {str(e)}. Please try again with a different request."
                })
                st.rerun()

if __name__ == "__main__":
    main()