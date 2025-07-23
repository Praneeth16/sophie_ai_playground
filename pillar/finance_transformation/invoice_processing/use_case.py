import streamlit as st
import tempfile
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from llama_cloud_services import LlamaParse
import pandas as pd

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

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'parsed_results' not in st.session_state:
        st.session_state.parsed_results = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'parsing_complete' not in st.session_state:
        st.session_state.parsing_complete = False

def validate_api_key() -> bool:
    """Validate LlamaCloud API key"""
    if not LLAMACLOUD_API_KEY or LLAMACLOUD_API_KEY == "your-llamacloud-api-key":
        return False
    
    # Additional validation could be added here to test the key
    # For now, just check if it starts with expected format
    if not LLAMACLOUD_API_KEY.startswith("llx-"):
        return False
    
    return True

def render_api_key_error():
    """Render API key configuration error message"""
    st.error("**Configuration Required:** LlamaCloud API Key is missing or invalid.")
    
    with st.expander("Setup Instructions", expanded=True):
        st.markdown("""
        ### How to get your LlamaCloud API Key:
        
        1. **Sign up** for a free account at [LlamaCloud](https://cloud.llamaindex.ai)
        2. **Navigate** to the API Keys section in your dashboard
        3. **Generate** a new API key
        4. **Copy** the key (it should start with 'llx-')
        
        ### Add the key to your Streamlit secrets:
        
        Create or edit `.streamlit/secrets.toml` in your project root:
        
        ```toml
        LLAMACLOUD_API_KEY = "llx-your-actual-api-key-here"
        ```
        
        ### Free Tier Information:
        - Parse up to **1,000 pages per day** for free
        - Supports **50+ file formats** (PDF, DOCX, PPTX, images, etc.)
        - Advanced parsing modes available
        
        Once configured, refresh this page to start parsing documents!
        """)
    
    st.info("Need help? Check the [LlamaCloud Documentation](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) for more details.")

def parse_document_with_llamacloud(uploaded_file, parsing_options) -> Optional[Dict[str, Any]]:
    """Parse document using LlamaCloud Parse API with advanced options"""
    try:
        # Build parser configuration based on options
        parser_config = {
            "api_key": LLAMACLOUD_API_KEY,
            "result_type": "markdown",
            "verbose": True,
            "language": "en"
        }
        
        # Add advanced parsing options
        if parsing_options.get("extract_charts"):
            parser_config["extract_charts"] = True
            parser_config["auto_mode"] = True
            parser_config["auto_mode_trigger_on_image_in_page"] = True
            parser_config["auto_mode_trigger_on_table_in_page"] = True
        
        if parsing_options.get("output_tables_as_html"):
            parser_config["output_tables_as_HTML"] = True
        
        if parsing_options.get("target_pages"):
            parser_config["target_pages"] = parsing_options["target_pages"]
        
        # Set parsing mode through preset or manual configuration
        parsing_mode = parsing_options.get("parsing_mode", "balanced")
        if parsing_mode == "fast":
            parser_config["fast_mode"] = True
        elif parsing_mode == "premium":
            parser_config["premium_mode"] = True
        # balanced is the default
        
        # Initialize LlamaParse client
        parser = LlamaParse(**parser_config)
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Parse the document
        result = parser.parse(tmp_file_path)
        
        # Extract different formats
        markdown_docs = result.get_markdown_documents(split_by_page=True)
        text_docs = result.get_text_documents(split_by_page=True)
        
        # Get image documents if available
        try:
            image_docs = result.get_image_documents(
                include_screenshot_images=True,
                include_object_images=True
            )
            image_count = len(image_docs) if image_docs else 0
        except:
            image_count = 0
        
        # Combine results
        parsed_data = {
            "markdown_content": "\n\n---\n\n".join([doc.text for doc in markdown_docs]) if markdown_docs else "",
            "text_content": "\n\n---\n\n".join([doc.text for doc in text_docs]) if text_docs else "",
            "raw_result": result,
            "page_count": len(markdown_docs) if markdown_docs else 0,
            "image_count": image_count,
            "parsing_options": parsing_options,
            "timestamp": datetime.now().isoformat()
        }
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return parsed_data
        
    except Exception as e:
        # Clean up temporary file on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        st.error(f"Error parsing document: {str(e)}")
        return None

def render_header():
    """Render page header with consistent styling"""
    st.markdown('<div class="hero-title">Invoice Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Advanced AI-powered document parsing</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-description">
    Upload PDF or image files to extract structured data using state-of-the-art AI parsing technology.
    View results in multiple formats and download processed content.
    </div>
    """, unsafe_allow_html=True)

def render_file_upload():
    """Render file upload interface"""
    st.markdown('<div class="arsenal-title">Upload Document</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP"
    )
    
    return uploaded_file

def render_parsing_options():
    """Render parsing configuration options"""
    with st.expander("Advanced Parsing Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            parsing_mode = st.selectbox(
                "Parsing Mode",
                ["fast", "balanced", "premium"],
                index=1,
                help="Fast: Quick processing, Premium: Maximum accuracy"
            )
            
            target_pages = st.text_input(
                "Target Pages (optional)",
                placeholder="e.g., 0,1,2 or 0-5",
                help="Specify pages to parse (0-indexed). Leave empty for all pages."
            )
        
        with col2:
            extract_charts = st.checkbox(
                "Extract Charts & Graphs",
                value=True,
                help="Enable extraction of visual elements like charts and diagrams"
            )
            
            output_tables_as_html = st.checkbox(
                "Output Tables as HTML",
                value=False,
                help="Render tables as HTML instead of markdown for better formatting"
            )
    
    return {
        "parsing_mode": parsing_mode,
        "target_pages": target_pages if target_pages else None,
        "extract_charts": extract_charts,
        "output_tables_as_html": output_tables_as_html
    }

def render_results_tabs(parsed_data: Dict[str, Any]):
    """Render results in tabbed interface"""
    st.markdown('<div class="arsenal-title">Parsing Results</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Markdown Output", "Text Output", "Document Info", "Raw JSON"])
    
    with tab1:
        st.markdown("### Markdown Format")
        
        if parsed_data["markdown_content"]:
            st.markdown(parsed_data["markdown_content"])
            
            # Download button
            st.download_button(
                label="Download Markdown",
                data=parsed_data["markdown_content"],
                file_name=f"parsed_{st.session_state.uploaded_file_name.rsplit('.', 1)[0]}.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.warning("No markdown content available.")
    
    with tab2:
        st.markdown("### Plain Text Format")
        st.markdown("Clean text without formatting, ideal for text analysis.")
        
        if parsed_data["text_content"]:
            st.text_area(
                "Text Output",
                value=parsed_data["text_content"],
                height=400,
                label_visibility="collapsed"
            )
            
            # Download button
            st.download_button(
                label="Download Text",
                data=parsed_data["text_content"],
                file_name=f"parsed_{st.session_state.uploaded_file_name.rsplit('.', 1)[0]}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.warning("No text content available.")
    
    with tab3:
        st.markdown("### Document Information")
        
        # Document metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="pillar-description">
                <strong>File Name:</strong><br>{st.session_state.uploaded_file_name}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="pillar-description">
                <strong>Page Count:</strong><br>{parsed_data["page_count"]} pages
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            parsed_time = datetime.fromisoformat(parsed_data["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"""
            <div class="pillar-description">
                <strong>Parsed At:</strong><br>{parsed_time}
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metadata row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="pillar-description">
                <strong>Images Found:</strong><br>{parsed_data.get("image_count", 0)} images
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            parsing_mode = parsed_data.get("parsing_options", {}).get("parsing_mode", "balanced")
            st.markdown(f"""
            <div class="pillar-description">
                <strong>Parsing Mode:</strong><br>{parsing_mode.title()}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            extract_charts = parsed_data.get("parsing_options", {}).get("extract_charts", False)
            charts_status = "Enabled" if extract_charts else "Disabled"
            st.markdown(f"""
            <div class="pillar-description">
                <strong>Chart Extraction:</strong><br>{charts_status}
            </div>
            """, unsafe_allow_html=True)
        
        # Character counts
        st.markdown("#### Content Statistics")
        markdown_chars = len(parsed_data["markdown_content"])
        text_chars = len(parsed_data["text_content"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Markdown Characters", f"{markdown_chars:,}")
        with col2:
            st.metric("Text Characters", f"{text_chars:,}")
        
        # Parsing options used
        st.markdown("#### Parsing Configuration")
        options = parsed_data.get("parsing_options", {})
        
        config_data = {
            "Parsing Mode": options.get("parsing_mode", "balanced").title(),
            "Extract Charts": "Yes" if options.get("extract_charts") else "No",
            "HTML Tables": "Yes" if options.get("output_tables_as_html") else "No",
            "Target Pages": options.get("target_pages") or "All pages"
        }
        
        config_df = pd.DataFrame(list(config_data.items()), columns=["Setting", "Value"])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    with tab4:
        st.markdown("### Raw JSON Data")
        st.markdown("Complete parsing results for advanced users and developers.")
        
        # Create simplified JSON for display (excluding large objects)
        display_data = {
            "file_name": st.session_state.uploaded_file_name,
            "page_count": parsed_data["page_count"],
            "timestamp": parsed_data["timestamp"],
            "markdown_length": len(parsed_data["markdown_content"]),
            "text_length": len(parsed_data["text_content"]),
            "has_markdown": bool(parsed_data["markdown_content"]),
            "has_text": bool(parsed_data["text_content"])
        }
        
        json_str = json.dumps(display_data, indent=2)
        st.code(json_str, language="json")
        
        # Download button for full results
        full_results = {
            "metadata": display_data,
            "markdown_content": parsed_data["markdown_content"],
            "text_content": parsed_data["text_content"]
        }
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(full_results, indent=2),
            file_name=f"parsed_{st.session_state.uploaded_file_name.rsplit('.', 1)[0]}.json",
            mime="application/json",
            use_container_width=True
        )

def render_features_overview():
    """Render features overview section"""
    st.markdown('<div class="arsenal-title">What This Tool Can Do</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="pillar-description">
            <strong>Multiple Formats</strong><br>
            Supports PDF, PNG, JPG, JPEG, TIFF, and BMP files with high accuracy parsing.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pillar-description">
            <strong>Advanced Extraction</strong><br>
            Extracts tables, charts, text, and maintains document structure with AI precision.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pillar-description">
            <strong>Export Options</strong><br>
            Download results as Markdown, plain text, or JSON for further processing.
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Check API configuration
    if not validate_api_key():
        render_api_key_error()
        st.stop()
    
    st.divider()
    
    # Features overview
    render_features_overview()
    
    st.divider()
    
    # File upload section
    uploaded_file = render_file_upload()
    
    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Show file info
        file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
        st.success(f"File uploaded: {uploaded_file.name} ({file_size:.2f} MB)")
        
        # Parsing options
        parsing_options = render_parsing_options()
        
        st.divider()
        
        # Parse button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Parse Document", type="primary", use_container_width=True):
                with st.spinner("Parsing document....."):
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Initializing parser...")
                    progress_bar.progress(25)
                    
                    status_text.text("Uploading and processing document...")
                    progress_bar.progress(50)
                    
                    # Parse document
                    parsed_results = parse_document_with_llamacloud(uploaded_file, parsing_options)
                    
                    if parsed_results:
                        status_text.text("Parsing complete!")
                        progress_bar.progress(100)
                        
                        st.session_state.parsed_results = parsed_results
                        st.session_state.parsing_complete = True
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success("Document parsed successfully!")
                        st.rerun()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("Failed to parse document. Please try again.")
    else:
        # Show demo section only when no file is uploaded
        st.divider()
    
    # Show results if available
    if st.session_state.parsing_complete and st.session_state.parsed_results:
        st.divider()
        render_results_tabs(st.session_state.parsed_results)
        
        # Clear results button
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Upload New Document", use_container_width=True):
                # Clear session state
                st.session_state.parsed_results = None
                st.session_state.uploaded_file_name = None
                st.session_state.parsing_complete = False
                st.rerun()

if __name__ == "__main__":
    main() 