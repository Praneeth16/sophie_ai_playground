import streamlit as st
import tempfile
import os
from datetime import datetime
from typing import Optional, Dict, Any
from llama_cloud_services import LlamaParse
from openai import OpenAI

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

# Configuration
LLAMACLOUD_API_KEY = st.secrets.get("LLAMACLOUD_API_KEY", "your-llamacloud-api-key")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "your-openrouter-api-key")

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'job_description_text' not in st.session_state:
        st.session_state.job_description_text = ""
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'parsing_complete' not in st.session_state:
        st.session_state.parsing_complete = False
    if 'boolean_results' not in st.session_state:
        st.session_state.boolean_results = {}

def validate_api_keys() -> tuple[bool, str]:
    """Validate required API keys"""
    if not LLAMACLOUD_API_KEY or LLAMACLOUD_API_KEY == "your-llamacloud-api-key":
        return False, "LlamaCloud API Key is missing or invalid."
    
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key":
        return False, "OpenRouter API Key is missing or invalid."
    
    # Additional validation for key format
    if not LLAMACLOUD_API_KEY.startswith("llx-"):
        return False, "LlamaCloud API Key format is invalid."
    
    return True, ""

def render_api_key_error(error_message: str):
    """Render API key configuration error message"""
    st.error(f"**Configuration Required:** {error_message}")
    
    with st.expander("Setup Instructions", expanded=True):
        st.markdown("""
        ### Required API Keys:
        
        #### 1. LlamaCloud API Key
        - **Sign up** for a free account at [LlamaCloud](https://cloud.llamaindex.ai)
        - **Navigate** to the API Keys section in your dashboard
        - **Generate** a new API key (should start with 'llx-')
        
        #### 2. OpenRouter API Key
        - **Sign up** for an account at [OpenRouter](https://openrouter.ai)
        - **Navigate** to the API Keys section
        - **Generate** a new API key
        
        ### Add keys to your Streamlit secrets:
        
        Create or edit `.streamlit/secrets.toml` in your project root:
        
        ```toml
        LLAMACLOUD_API_KEY = "llx-your-llamacloud-key-here"
        OPENROUTER_API_KEY = "your-openrouter-key-here"
        ```
        
        Once configured, refresh this page to start using the tool!
        """)

def parse_job_description_pdf(uploaded_file) -> Optional[str]:
    """Parse job description PDF using LlamaCloud Parse API"""
    try:
        # Initialize LlamaParse client
        parser = LlamaParse(
            api_key=LLAMACLOUD_API_KEY,
            result_type="text",
            verbose=False,
            language="en"
        )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Parse the document
        result = parser.parse(tmp_file_path)
        
        # Extract text content
        text_docs = result.get_text_documents()
        extracted_text = "\n\n".join([doc.text for doc in text_docs]) if text_docs else ""
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return extracted_text
        
    except Exception as e:
        # Clean up temporary file on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        st.error(f"Error parsing PDF: {str(e)}")
        return None

def generate_boolean_strings(job_description: str) -> Dict[str, str]:
    """Generate platform-specific boolean search strings using OpenRouter API"""
    
    # Platform-specific prompts
    platform_prompts = {
        "LinkedIn": """You are an expert LinkedIn talent sourcer. Create a LinkedIn-optimized boolean search string from this job description.

LinkedIn Boolean Search Guidelines:
1. Use LinkedIn's boolean operators: AND, OR, NOT, quotation marks for exact phrases
2. Focus on job titles, skills, and experience levels most relevant to LinkedIn profiles
3. Include industry-specific keywords and synonyms
4. Use LinkedIn-friendly terms (e.g., "Software Engineer" OR "Software Developer")
5. Consider LinkedIn's character limit and search behavior
6. Include location terms if specified
7. Use parentheses for grouping

Format: Return ONLY the boolean string, no explanations or code blocks.
Maximum 500 characters.""",

        "Naukri": """You are an expert Naukri.com talent sourcer. Create a Naukri-optimized boolean search string from this job description.

Naukri Boolean Search Guidelines:
1. Use Naukri's boolean operators: AND, OR, NOT, quotation marks for exact phrases
2. Focus on Indian job market terminology and skills
3. Include relevant certifications and educational qualifications common in India
4. Use Indian city names and regional terms
5. Consider experience levels in Indian context (fresher, experienced, etc.)
6. Include both Indian and international technology terms
7. Focus on skills sections and job titles popular on Naukri

Format: Return ONLY the boolean string, no explanations or code blocks.
Maximum 500 characters.""",

        "Indeed": """You are an expert Indeed talent sourcer. Create an Indeed-optimized boolean search string from this job description.

Indeed Boolean Search Guidelines:
1. Use Indeed's boolean operators: AND, OR, NOT, quotation marks for exact phrases
2. Focus on job titles and skills that work well across Indeed's global database
3. Include both technical and soft skills relevant to the role
4. Use common industry terminology and abbreviations
5. Consider Indeed's resume parsing and keyword matching
6. Include experience ranges and education levels
7. Use location-neutral terms unless specific geography is mentioned

Format: Return ONLY the boolean string, no explanations or code blocks.
Maximum 500 characters.""",

        "General": """You are an expert boolean search specialist. Create a comprehensive boolean search string from this job description.

General Boolean Search Guidelines:
1. Extract core technical skills, tools, and technologies
2. Identify key job titles and synonyms
3. Include experience level indicators
4. Add relevant certifications and education
5. Use proper boolean operators: AND, OR, NOT, parentheses
6. Group similar terms with OR, combine requirements with AND
7. Exclude unwanted terms with NOT
8. Keep it comprehensive but concise

Format: Return ONLY the boolean string, no explanations or code blocks.
Maximum 500 characters."""
    }
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    
    results = {}
    
    for platform, prompt in platform_prompts.items():
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://hr-copilot.streamlit.app",
                    "X-Title": "HR CoPilot - Boolean Search Generator",
                },
                model="google/gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Job Description:\n\n{job_description}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            boolean_string = completion.choices[0].message.content.strip()
            results[platform] = boolean_string
            
        except Exception as e:
            results[platform] = f"Error generating {platform} boolean string: {str(e)}"
    
    return results

def render_header():
    """Render page header with consistent styling"""
    st.markdown('<div class="hero-title">Boolean Search Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">AI-powered boolean strings for every platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-description">
    Upload job description PDFs or paste text to generate optimized boolean search strings 
    for LinkedIn, Naukri, Indeed, and other platforms. Find candidates faster with AI-crafted search queries.
    </div>
    """, unsafe_allow_html=True)

def render_input_section():
    """Render job description input section"""
    st.markdown('<div class="arsenal-title">Job Description Input</div>', unsafe_allow_html=True)
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload PDF", "Paste Text"],
        horizontal=True,
        help="Upload a PDF file or paste job description text directly"
    )
    
    job_description = ""
    
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF containing the job description"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Show file info
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
            st.success(f"File uploaded: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Parse button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Extract Text from PDF", type="primary", use_container_width=True):
                    with st.spinner("Extracting text from PDF..."):
                        extracted_text = parse_job_description_pdf(uploaded_file)
                        
                        if extracted_text:
                            st.session_state.job_description_text = extracted_text
                            st.session_state.parsing_complete = True
                            st.success("Text extracted successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to extract text from PDF. Please try again.")
            
            # Display extracted text if available
            if st.session_state.parsing_complete and st.session_state.job_description_text:
                with st.expander("Extracted Text (Click to view/edit)", expanded=False):
                    job_description = st.text_area(
                        "Extracted Job Description:",
                        value=st.session_state.job_description_text,
                        height=300,
                        key="extracted_text_area"
                    )
                    if st.button("Update Text", key="update_extracted"):
                        st.session_state.job_description_text = job_description
                        st.rerun()
    
    else:  # Paste Text
        job_description = st.text_area(
            "Paste Job Description:",
            height=300,
            placeholder="Paste the complete job description here...",
            help="Copy and paste the job description text"
        )
    
    return job_description

def render_boolean_results(boolean_results: Dict[str, str]):
    """Render boolean search results in tabbed interface"""
    st.markdown('<div class="arsenal-title">Boolean Search Strings</div>', unsafe_allow_html=True)
    
    # Create tabs for different platforms
    tab1, tab2, tab3, tab4 = st.tabs(["LinkedIn", "Naukri", "Indeed", "General"])
    
    platforms = ["LinkedIn", "Naukri", "Indeed", "General"]
    tabs = [tab1, tab2, tab3, tab4]
    
    platform_descriptions = {
        "LinkedIn": "Optimized for LinkedIn's search algorithm and professional networking context",
        "Naukri": "Tailored for Indian job market with relevant terminology and skills",
        "Indeed": "Designed for Indeed's global database and resume parsing system", 
        "General": "Comprehensive boolean string suitable for most job boards and search engines"
    }
    
    for platform, tab in zip(platforms, tabs):
        with tab:
            st.markdown(f"### {platform} Search String")
            st.markdown(f"*{platform_descriptions[platform]}*")
            
            if platform in boolean_results:
                boolean_string = boolean_results[platform]
                
                if not boolean_string.startswith("Error"):
                    st.success("Boolean string generated successfully!")
                    
                    # Display the boolean string
                    st.code(boolean_string, language="text")
                    
                    # Copy button (using download as copy alternative)
                    st.download_button(
                        label=f"Download {platform} Boolean String",
                        data=boolean_string,
                        file_name=f"{platform.lower()}_boolean_search.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    # Character count
                    char_count = len(boolean_string)
                    color = "green" if char_count <= 500 else "orange" if char_count <= 600 else "red"
                    st.markdown(f"**Character count:** <span style='color: {color}'>{char_count}</span>", unsafe_allow_html=True)
                    
                    # Usage tips
                    with st.expander(f"How to use this {platform} search", expanded=False):
                        if platform == "LinkedIn":
                            st.markdown("""
                            **LinkedIn Search Tips:**
                            1. Go to LinkedIn's People search
                            2. Paste the boolean string in the search box
                            3. Use additional filters (location, company, etc.)
                            4. Save the search for future use
                            5. Consider connection levels and premium features
                            """)
                        elif platform == "Naukri":
                            st.markdown("""
                            **Naukri Search Tips:**
                            1. Use the Advanced Search option
                            2. Paste the boolean string in the keyword field
                            3. Add location and experience filters
                            4. Use the boolean string in resume database search
                            5. Consider functional areas and industries
                            """)
                        elif platform == "Indeed":
                            st.markdown("""
                            **Indeed Search Tips:**
                            1. Use Indeed's resume search (for employers)
                            2. Paste the boolean string in the keyword field
                            3. Combine with location and date filters
                            4. Try variations of the search string
                            5. Use quotes for exact phrase matching
                            """)
                        else:
                            st.markdown("""
                            **General Search Tips:**
                            1. Test the string on different platforms
                            2. Modify based on platform-specific features
                            3. Use parentheses to group related terms
                            4. Adjust operator precedence as needed
                            5. Consider platform character limits
                            """)
                else:
                    st.error(boolean_string)
            else:
                st.warning(f"No {platform} boolean string generated yet.")

def render_features_overview():
    """Render features overview section"""
    st.markdown('<div class="arsenal-title">What This Tool Can Do</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="pillar-description">
            <strong>Smart PDF Processing</strong><br>
            Upload job description PDFs and extract text with AI-powered parsing technology.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pillar-description">
            <strong>Platform Optimization</strong><br>
            Generate boolean strings optimized for LinkedIn, Naukri, Indeed, and other platforms.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pillar-description">
            <strong>Advanced Boolean Logic</strong><br>
            AI-crafted search strings with proper operators, grouping, and platform-specific terminology.
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Check API configuration
    is_valid, error_message = validate_api_keys()
    if not is_valid:
        render_api_key_error(error_message)
        st.stop()
    
    st.divider()
    
    # Features overview
    render_features_overview()
    
    st.divider()
    
    # Input section
    job_description = render_input_section()
    
    # Generate boolean strings section
    if job_description and job_description.strip():
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Generate Boolean Strings", type="primary", use_container_width=True):
                with st.spinner("Generating platform-specific boolean search strings..."):
                    boolean_results = generate_boolean_strings(job_description)
                    st.session_state.boolean_results = boolean_results
                    st.success("Boolean strings generated successfully!")
                    st.rerun()
    
    # Show results if available
    if st.session_state.boolean_results:
        st.divider()
        render_boolean_results(st.session_state.boolean_results)
        
        # Clear results button
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Start New Search", use_container_width=True):
                # Clear session state
                st.session_state.job_description_text = ""
                st.session_state.uploaded_file_name = None
                st.session_state.parsing_complete = False
                st.session_state.boolean_results = {}
                st.rerun()

if __name__ == "__main__":
    main()
