import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import io
from typing import List, Dict
from pydantic import BaseModel

load_dotenv()

# Pydantic models for structured output
class LanguageDetection(BaseModel):
    language_code: str
    language_name: str
    confidence: float

class Translation(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str

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
        Detect the language of the given text using structured output
        """
        try:
            # Language mapping for better detection
            language_info = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
                'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean',
                'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi', 'nl': 'Dutch',
                'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish',
                'pl': 'Polish', 'tr': 'Turkish', 'th': 'Thai', 'vi': 'Vietnamese'
            }
            
            system_prompt = f"""You are a language detection expert. Analyze the given text and identify its language.
            
            Return the response in the following format:
            - language_code: ISO 639-1 language code (e.g., 'en', 'es', 'fr')
            - language_name: Full language name (e.g., 'English', 'Spanish', 'French')
            - confidence: Confidence score between 0.0 and 1.0
            
            Supported languages: {', '.join([f'{code} ({name})' for code, name in language_info.items()])}"""

            response = self.client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Detect the language of this text: {text[:1000]}"}
                ],
                response_format={"type": "json_schema", "json_schema": {"name": "language_detection", "schema": LanguageDetection.model_json_schema()}},
                extra_headers={
                    "HTTP-Referer": "https://hr-copilot.streamlit.app",
                    "X-Title": "Invoice Translation Tool",
                }
            )
            
            result = LanguageDetection.model_validate_json(response.choices[0].message.content)
            return result.language_code, result.language_name, result.confidence
            
        except Exception as e:
            # Fallback to simple detection if structured output fails
            try:
                response = self.client.chat.completions.create(
                    model="google/gemini-2.5-flash-lite",
                    messages=[
                        {"role": "system", "content": "You are a language detection expert. Respond with only the ISO 639-1 language code."},
                        {"role": "user", "content": f"Detect the language of this text and respond with only the ISO 639-1 language code: {text[:1000]}"}
                    ],
                    extra_headers={
                        "HTTP-Referer": "https://hr-copilot.streamlit.app",
                        "X-Title": "Invoice Translation Tool",
                    }
                )
                detected_code = response.choices[0].message.content.strip().lower()
                return detected_code, "Unknown", 0.8
            except Exception as fallback_e:
                raise Exception(f"Error detecting language: {str(e)} (Fallback error: {str(fallback_e)})")

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

    def translate_text(self, text, target_language_code, target_language_name=None):
        """
        Translate text from any language to any target language using structured output
        """
        try:
            # First detect the source language
            source_code, source_name, confidence = self.detect_language(text)
            
            # If source and target are the same, return as is
            if source_code.lower() == target_language_code.lower():
                return text, source_code, source_name, confidence

            # Language mapping for better prompts
            language_names = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
                'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean',
                'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi', 'nl': 'Dutch',
                'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish',
                'pl': 'Polish', 'tr': 'Turkish', 'th': 'Thai', 'vi': 'Vietnamese'
            }
            
            source_lang_name = language_names.get(source_code, source_name)
            target_lang_name = target_language_name or language_names.get(target_language_code, target_language_code)
            
            system_prompt = f"""You are a professional translator. 
            Your task is to translate the given text from {source_lang_name} to {target_lang_name} while:
            1. Maintaining the exact meaning and context
            2. Using natural, fluent {target_lang_name}
            3. Preserving any formatting or structure
            4. Keeping numbers and special characters intact
            5. Preserving all the numbers strictly without changing the commas and decimals
            6. Do not add any markdown formatting or code blocks
            
            Return the response in the following format:
            - original_text: The original input text
            - translated_text: The translated text
            - source_language: Source language code
            - target_language: Target language code"""

            try:
                # Try structured output first
                response = self.client.chat.completions.create(
                    model="google/gemini-2.5-flash-lite",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate this text from {source_lang_name} to {target_lang_name}:\n\n{text}"}
                    ],
                    response_format={"type": "json_schema", "json_schema": {"name": "translation", "schema": Translation.model_json_schema()}},
                    extra_headers={
                        "HTTP-Referer": "https://hr-copilot.streamlit.app",
                        "X-Title": "Invoice Translation Tool",
                    }
                )
                
                result = Translation.model_validate_json(response.choices[0].message.content)
                translated_text = self.clean_translation_output(result.translated_text)
                
            except Exception:
                # Fallback to simple translation if structured output fails
                simple_prompt = f"""You are a professional translator. 
                Translate the following text from {source_lang_name} to {target_lang_name}.
                Maintain exact meaning, preserve formatting and numbers.
                Respond with only the translated text, no additional formatting."""
                
                response = self.client.chat.completions.create(
                    model="google/gemini-2.5-flash-lite",
                    messages=[
                        {"role": "system", "content": simple_prompt},
                        {"role": "user", "content": text}
                    ],
                    extra_headers={
                        "HTTP-Referer": "https://hr-copilot.streamlit.app",
                        "X-Title": "Invoice Translation Tool",
                    }
                )
                
                translated_text = self.clean_translation_output(response.choices[0].message.content)
            
            return translated_text, source_code, source_name, confidence
            
        except Exception as e:
            raise Exception(f"Error translating text: {str(e)}")

    def translate_batch(self, texts: List[str], target_language_code: str, target_language_name: str) -> List[Dict]:
        """
        Translate a batch of texts to target language
        """
        results = []
        
        for text in texts:
            if pd.isna(text) or str(text).strip() == '':
                results.append({
                    'translated_text': '',
                    'source_code': '',
                    'source_name': '',
                    'confidence': 0.0
                })
            else:
                try:
                    translated, source_code, source_name, confidence = self.translate_text(
                        str(text), target_language_code, target_language_name
                    )
                    results.append({
                        'translated_text': translated,
                        'source_code': source_code,
                        'source_name': source_name,
                        'confidence': confidence
                    })
                except Exception as e:
                    st.warning(f"Failed to translate text: {str(text)[:50]}... Error: {str(e)}")
                    results.append({
                        'translated_text': str(text),  # Keep original if translation fails
                        'source_code': 'unknown',
                        'source_name': 'Unknown',
                        'confidence': 0.0
                    })
        
        return results

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'translation_result' not in st.session_state:
        st.session_state.translation_result = None
    if 'detected_language' not in st.session_state:
        st.session_state.detected_language = None
    if 'original_text' not in st.session_state:
        st.session_state.original_text = ""
    if 'batch_mode' not in st.session_state:
        st.session_state.batch_mode = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'batch_result' not in st.session_state:
        st.session_state.batch_result = None

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
    st.markdown('<div class="hero-subtitle">Translate documents individually or process batch files</div>', unsafe_allow_html=True)

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

def render_single_translation_results(translated_text, source_code, source_name, target_code, target_name, confidence, original_text):
    """Render the translation results for single document mode"""
    st.markdown('<div class="arsenal-title">Translation Results</div>', unsafe_allow_html=True)
    
    # Language detection and translation info
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], border=True)
    with col1:
        st.metric("Source Language", f"{source_name} ({source_code.upper()})")

    with col2:
        st.metric("Target Language", f"{target_name} ({target_code.upper()})")

    with col3:
        st.metric("Confidence", f"{confidence:.2f}")
        
    with col4:
        if source_code.lower() == target_code.lower():
            st.metric("Translation Status", "Not Required")
        else:
            st.metric("Translation Status", "Completed")
    
    # Show results
    if source_code.lower() == target_code.lower():
        st.info(f"Document is already in {target_name}. No translation required.")
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
            st.markdown(f"**Original Text ({source_name}):**")
            with st.container(border=True):
                st.markdown(f"{original_text}")
        
        with col2:
            st.markdown(f"**Translated Text ({target_name}):**")
            with st.container(border=True):
                st.markdown(f"""{translated_text}""")

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
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["Single Document", "Batch Mode"])
    
    with tab1:
        handle_single_document_mode(translator)
    
    with tab2:
        handle_batch_mode(translator)

def handle_single_document_mode(translator):
    """Handle single document translation mode"""
    # Render input form
    document_text = render_input_form()
    
    # Target language selection for single document
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Target Language:**")
        languages = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it',
            'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja', 'Korean': 'ko',
            'Chinese': 'zh', 'Arabic': 'ar', 'Hindi': 'hi', 'Dutch': 'nl',
            'Swedish': 'sv', 'Danish': 'da', 'Norwegian': 'no', 'Finnish': 'fi',
            'Polish': 'pl', 'Turkish': 'tr', 'Thai': 'th', 'Vietnamese': 'vi'
        }
        
        target_language_name = st.selectbox(
            "Select target language:",
            options=list(languages.keys()),
            index=0,  # Default to English
            help="Choose the language you want to translate the text to",
            key="single_target_lang"
        )
        
        target_language_code = languages[target_language_name]
    
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
                    translated_text, source_code, source_name, confidence = translator.translate_text(
                        document_text, target_language_code, target_language_name
                    )
                    
                    progress_bar.progress(75, text="Completing translation...")
                    
                    # Store results in session state
                    st.session_state.translation_result = translated_text
                    st.session_state.detected_language = source_code
                    st.session_state.detected_language_name = source_name
                    st.session_state.target_language = target_language_code
                    st.session_state.target_language_name = target_language_name
                    st.session_state.confidence = confidence
                    st.session_state.original_text = document_text
                    
                    progress_bar.progress(100, text="Translation complete!")
                    progress_bar.empty()
                    
                    if source_code.lower() == target_language_code.lower():
                        st.info(f"Document is already in {target_language_name}. No translation required.")
                    else:
                        st.success("Translation completed successfully!")
                    
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")
                    return
    
    # Show results if available
    if (st.session_state.translation_result and 
        st.session_state.detected_language and 
        hasattr(st.session_state, 'target_language')):
        st.markdown("---")
        render_single_translation_results(
            st.session_state.translation_result,
            st.session_state.detected_language,
            st.session_state.detected_language_name,
            st.session_state.target_language,
            st.session_state.target_language_name,
            st.session_state.confidence,
            st.session_state.original_text
        )
        
        # Download option
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.detected_language.lower() != st.session_state.target_language.lower():
                st.download_button(
                    label="Download Translated Text",
                    data=st.session_state.translation_result,
                    file_name=f"translated_document_{st.session_state.detected_language}_to_{st.session_state.target_language}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

def handle_batch_mode(translator):
    """Handle batch translation mode"""
    # File upload
    df, uploaded_file = render_batch_upload()
    
    if df is not None and uploaded_file is not None:
        st.markdown("---")
        
        # Configuration
        selected_column, target_language_code, target_language_name, new_column_name = render_batch_configuration(df)
        
        # Translation button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Translate Column", type="primary", use_container_width=True):
                if not selected_column:
                    st.warning("Please select a column to translate.")
                    return
                
                if not new_column_name.strip():
                    st.warning("Please provide a name for the new translated column.")
                    return
                
                # Show progress
                with st.spinner(f"Translating {len(df)} rows to {target_language_name}..."):
                    try:
                        progress_bar = st.progress(0)
                        progress_bar.progress(10, text="Preparing translation...")
                        
                        # Get texts to translate
                        texts_to_translate = df[selected_column].tolist()
                        
                        progress_bar.progress(25, text="Starting translation...")
                        
                        # Translate batch
                        total_texts = len(texts_to_translate)
                        
                        # Use the new batch translation method
                        progress_bar.progress(50, text="Translating batch...")
                        translation_results = translator.translate_batch(
                            texts_to_translate, target_language_code, target_language_name
                        )
                        
                        progress_bar.progress(95, text="Finalizing results...")
                        
                        # Extract translated texts and metadata
                        translated_texts = [result['translated_text'] for result in translation_results]
                        source_languages = [result['source_code'] for result in translation_results]
                        
                        # Create result dataframe
                        result_df = df.copy()
                        result_df[new_column_name] = translated_texts
                        result_df[f"{new_column_name}_source_lang"] = source_languages
                        
                        # Store results in session state
                        st.session_state.batch_result = result_df
                        st.session_state.uploaded_file = uploaded_file
                        
                        progress_bar.progress(100, text="Translation complete!")
                        progress_bar.empty()
                        
                        st.success(f"Batch translation completed! Translated {total_texts} rows to {target_language_name}.")
                        
                    except Exception as e:
                        st.error(f"Batch translation failed: {str(e)}")
                        return
        
        # Show results if available
        if st.session_state.batch_result is not None and st.session_state.uploaded_file is not None:
            st.markdown("---")
            render_batch_results(st.session_state.batch_result, st.session_state.uploaded_file.name)

def render_batch_upload():
    """Render the batch file upload interface"""
    st.markdown('<div class="arsenal-title">File Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file:",
        type=['xlsx', 'xls', 'csv'],
        help="Upload an Excel (.xlsx, .xls) or CSV file containing the data you want to translate",
        key="batch_file_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file based on its type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show preview
            with st.expander("Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            return df, uploaded_file
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None, None
    
    return None, None

def render_batch_configuration(df):
    """Render batch translation configuration"""
    st.markdown('<div class="arsenal-title">Translation Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Column selection
        selected_column = st.selectbox(
            "Select column to translate:",
            options=df.columns.tolist(),
            help="Choose the column containing text that you want to translate"
        )
    
    with col2:
        # Target language selection
        languages = {
            'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it',
            'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja', 'Korean': 'ko',
            'Chinese': 'zh', 'Arabic': 'ar', 'Hindi': 'hi', 'Dutch': 'nl',
            'Swedish': 'sv', 'Danish': 'da', 'Norwegian': 'no', 'Finnish': 'fi',
            'Polish': 'pl', 'Turkish': 'tr', 'Thai': 'th', 'Vietnamese': 'vi'
        }
        
        target_language_name = st.selectbox(
            "Select target language:",
            options=list(languages.keys()),
            help="Choose the language you want to translate the text to"
        )
        
        target_language_code = languages[target_language_name]
    
    # New column name (full width)
    new_column_name = st.text_input(
        "New column name:",
        value=f"{selected_column}_{target_language_name.lower()}" if selected_column else "",
        help="Name for the new column that will contain the translated text"
    )
    
    # Column Preview (after all configuration)
    if selected_column:
        st.markdown("---")
        st.markdown("**Column Preview:**")
        preview_data = df[selected_column].dropna().head(5).tolist()
        for i, item in enumerate(preview_data, 1):
            st.write(f"{i}. {str(item)[:100]}{'...' if len(str(item)) > 100 else ''}")
    
    return selected_column, target_language_code, target_language_name, new_column_name

def render_batch_results(result_df, original_filename):
    """Render batch translation results"""
    st.markdown('<div class="arsenal-title">Translation Results</div>', unsafe_allow_html=True)
    
    st.success(f"Translation completed! Processed {len(result_df)} rows.")
    
    # Show preview of results
    with st.expander("Preview Results", expanded=True):
        st.dataframe(result_df.head(10), use_container_width=True)
    
    # Download section
    st.markdown("### Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel download
        excel_buffer = io.BytesIO()
        result_df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)
        
        filename_base = original_filename.rsplit('.', 1)[0]
        excel_filename = f"{filename_base}_translated.xlsx"
        
        st.download_button(
            label="Download as Excel",
            data=excel_buffer.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        # CSV download
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        csv_filename = f"{filename_base}_translated.csv"
        
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()  