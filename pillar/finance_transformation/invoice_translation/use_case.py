import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Multi-Language Invoice Translation",
    page_icon="artifacts/mpg.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.logo(
    image="artifacts/mpg.jpeg",
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

# Load CSS
load_css()

# API Configuration
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")

# Initialize OpenRouter client
def get_openrouter_client():
    """Get configured OpenRouter client"""
    if OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        return None
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

class InvoiceTranslator:
    """Invoice translation service using OpenRouter API"""
    
    def __init__(self, client):
        self.client = client
    
    def detect_language(self, text):
        """
        Detect the language of the given text using OpenAI API
        """
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a language detection expert. Respond with only the ISO 639-1 language code."},
                    {"role": "user", "content": f"Detect the language of this text and respond with only the ISO 639-1 language code: {text[:1000]}"}
                ],
                extra_headers={
                    "HTTP-Referer": "https://hr-copilot.streamlit.app",
                    "X-Title": "Invoice Translation Tool",
                }
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error detecting language: {str(e)}")

    def clean_translation_output(self, text):
        """Clean up the translation output by removing unwanted markdown formatting"""
        # Remove markdown code block indicators
        text = text.replace("```markdown", "").replace("```", "")
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        # Remove any extra newlines at the beginning
        while text.startswith('\n'):
            text = text[1:]
            
        return text

    def translate_to_english(self, markdown_text):
        """
        Translate markdown text to English while preserving the markdown structure
        """
        try:
            # First detect the language
            source_language = self.detect_language(markdown_text)
            
            # If already English, return as is
            if source_language.lower() == 'en':
                return markdown_text, source_language

            # Prepare the translation prompt
            system_prompt = """You are a professional translator. 
            Your task is to translate the given text to English while:
            1. Maintaining the exact same structure and formatting
            2. Only translating the actual content
            3. Keeping all special characters, numbers, and formatting intact
            4. Preserving all the numbers strictly without changing the commas and decimals
            5. Do not add any markdown formatting or code blocks
            Respond with only the translated text, no additional formatting."""

            response = self.client.chat.completions.create(
                model="mistralai/mistral-medium-3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this text from {source_language} to English:\n\n{markdown_text}"}
                ],
                extra_headers={
                    "HTTP-Referer": "https://hr-copilot.streamlit.app",
                    "X-Title": "Invoice Translation Tool",
                }
            )
            
            # Clean the output
            translated_text = self.clean_translation_output(response.choices[0].message.content)
            
            return translated_text, source_language
        except Exception as e:
            raise Exception(f"Error translating text: {str(e)}")

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'translation_result' not in st.session_state:
        st.session_state.translation_result = None
    if 'detected_language' not in st.session_state:
        st.session_state.detected_language = None
    if 'original_text' not in st.session_state:
        st.session_state.original_text = ""

def validate_api_key():
    """Validate OpenRouter API key"""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        return False
    return True

def render_api_key_error():
    """Render API key configuration error message"""
    st.error("Configuration Required: OpenRouter API Key is missing or invalid.")
    
    with st.expander("Setup Instructions", expanded=True):
        st.markdown("""
        ### How to get your OpenRouter API Key:
        
        1. **Sign up** for an account at [OpenRouter](https://openrouter.ai)
        2. **Navigate** to the Keys section in your dashboard
        3. **Generate** a new API key
        4. **Copy** the key
        
        ### Add the key to your configuration:
        
        **Option 1: Streamlit Secrets** (Recommended)
        Create or edit `.streamlit/secrets.toml` in your project root:
        
        ```toml
        OPENROUTER_API_KEY = "your-actual-api-key-here"
        ```
        
        **Option 2: Environment Variable**
        Set the environment variable:
        
        ```bash
        export OPENROUTER_API_KEY="your-actual-api-key-here"
        ```
        
        ### Supported Models:
        - Language Detection: OpenAI GPT-4.1
        - Translation: Mistral Medium-3
        
        Once configured, refresh this page to start translating documents!
        """)
    
    st.info("Need help? Check the [OpenRouter Documentation](https://openrouter.ai/docs) for more details.")

def render_header():
    """Render the page header"""
    st.markdown('<div class="hero-title">Multi-Language Invoice Translation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Automatically detect and translate invoices from any language to English</div>', unsafe_allow_html=True)

def render_input_form():
    """Render the text input form"""
    st.markdown('<div class="arsenal-title">Document Text Input</div>', unsafe_allow_html=True)
    
    # Text input area
    document_text = st.text_area(
        "Paste your invoice or document text here:",
        height=300,
        placeholder="Copy and paste the text content from your invoice or document. The system will automatically detect the language and translate it to English if needed.",
        help="Supports all major languages. Paste the complete document text for best translation results.",
        key="document_input"
    )
    
    return document_text

def render_translation_results(translated_text, detected_language, original_text):
    """Render the translation results"""
    st.markdown('<div class="arsenal-title">Translation Results</div>', unsafe_allow_html=True)
    
    # Language detection info
    col1, col2, col3 = st.columns([1, 1, 1], border=True)
    with col1:
        st.metric("Detected Language", detected_language.upper() if detected_language else "Unknown")

    with col2:
        st.metric("Target Language", "EN")

    with col3:
        if detected_language and detected_language.lower() == 'en':
            st.metric("Translation Status", "Not Required")
        else:
            st.metric("Translation Status", "Completed")
    
    # Show results
    if detected_language and detected_language.lower() == 'en':
        st.info("Document is already in English. No translation required.")
        with st.container(border=True):
            st.markdown("**Original Text:**")
            st.markdown(
                f"""
                <div style="
                    height: 400px; 
                    overflow-y: auto; 
                    overflow-x: hidden; 
                    padding: 15px; 
                    background-color: #f8f9fa; 
                    border-radius: 8px;
                    border: 1px solid #e9ecef;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                ">
                {original_text}
                </div>
                """, 
                unsafe_allow_html=True
            )
    else:
        # Show original and translated side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Text:**")
            with st.container(border=True):
                st.markdown(f"{original_text}")
        
        with col2:
            st.markdown("**Translated Text (English):**")
            with st.container(border=True):
                st.markdown(f"""{translated_text}""")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Check API configuration
    if not validate_api_key():
        render_api_key_error()
        return
    
    # Initialize translator
    client = get_openrouter_client()
    if not client:
        st.error("Failed to initialize OpenRouter client. Please check your API key configuration.")
        return
    
    translator = InvoiceTranslator(client)
    
    # Render input form
    document_text = render_input_form()
    
    # Translation button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Translate Document", type="primary", use_container_width=True):
            if not document_text.strip():
                st.warning("Please paste some document text to translate.")
                return
            
            # Show progress
            with st.spinner("Detecting language and translating..."):
                try:
                    progress_bar = st.progress(0)
                    progress_bar.progress(25, text="Detecting language...")
                    
                    # Perform translation
                    translated_text, detected_language = translator.translate_to_english(document_text)
                    
                    progress_bar.progress(75, text="Completing translation...")
                    
                    # Store results in session state
                    st.session_state.translation_result = translated_text
                    st.session_state.detected_language = detected_language
                    st.session_state.original_text = document_text
                    
                    progress_bar.progress(100, text="Translation complete!")
                    progress_bar.empty()
                    
                    st.success("Translation completed successfully!")
                    
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")
                    return
    
    # Show results if available
    if st.session_state.translation_result and st.session_state.detected_language:
        st.markdown("---")
        render_translation_results(
            st.session_state.translation_result,
            st.session_state.detected_language,
            st.session_state.original_text
        )
        
        # Download option
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.detected_language.lower() != 'en':
                st.download_button(
                    label="Download Translated Text",
                    data=st.session_state.translation_result,
                    file_name=f"translated_document_{st.session_state.detected_language}_to_en.txt",
                    mime="text/plain",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()  