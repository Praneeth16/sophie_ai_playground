from sys import byteorder
import streamlit as st
import pandas as pd
import re
from openai import OpenAI
from rapidfuzz import fuzz
import json


# Configure the page
st.set_page_config(
    page_title="Decision Maker Discovery", 
    layout="wide",
    initial_sidebar_state="collapsed"
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
    
    # Additional custom CSS for this specific use case
    custom_css = """
    <style>
    /* Success and error message styling with light orange theme */
    .success-message {
        background: #fff7ed;
        color: #c2410c;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b35;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .error-message {
        background: #fee2e2;
        color: #991b1b;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .info-message {
        background: #fff7ed;
        color: #ea580c;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b35;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Upload area styling with light orange theme */
    .upload-container {
        background: #fff7ed;
        border: 2px dashed #ff6b35;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #ea580c;
        background: #fed7aa;
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

# Model configuration
MODEL_NAME = "moonshotai/kimi-k2"

# --------------------------
# HEADER DETECTION IN EXCEL
# --------------------------
KNOWN_DESIGNATIONS = ['Country','Franchise Country (Yes/No)','Notes about Country','Main POC for all general requests']

@st.cache_data(show_spinner=False)
def load_excel_with_dynamic_header(uploaded_file):
    df_raw = pd.read_excel(uploaded_file, header=None)
    header_row_idx = None
    for i in range(min(10, len(df_raw))):
        row = df_raw.iloc[i].astype(str)
        if any(any(key in cell for key in KNOWN_DESIGNATIONS) for cell in row):
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError("Header row not found in uploaded Excel.")
    df = pd.read_excel(uploaded_file, header=header_row_idx)
    df.columns = df.columns.str.strip().str.lower()
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip().str.lower()
    return df

# --------------------------
# PARSE COUNTRY + DESIGNATION
# --------------------------
def extract_country_designation(message):
    prompt = f"""
    Extract the country and the designation of the poc from this message.
    
    Once the country is identified, normalize it by mapping them to their most accurate and standardized using the below examples:

        Examples:
        "US" → "USA"
        "America" → "USA"
        "London" → "UK"
        "United Kingdom" → "UK"

        Any country in Middle East should be coded as "Middle East (UAE)"

    For identified designation, clean all special characters and respond in text only.

    Message: "{message}"

    Respond only in JSON:
    {{
    "Country": <standardised country>,
    "Designation": <designation>
    }}
        """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[ {"role": "user", "content": prompt} ]
    )

    output = json.loads(response.choices[0].message.content)
   
    country = output.get("Country", "")
    designation = output.get("Designation", "")

    return country, designation

# --------------------------
# GET CONTACT INFO FROM DF
# --------------------------
def get_contact_info(df, country, designation):
    country = country.strip().lower()
    designation = designation.strip().lower()

    best_col = None
    best_score = 0

    designation_cols = [col for col in df.columns if designation in col.lower()]
    if not designation_cols:
        # Fuzzy match: get best-matching column name
        for col in df.columns[1:]:  # Skip country/region column
            score = fuzz.token_sort_ratio(designation, col)
            if score > best_score:
                best_score = score
                best_col = col

        if best_score < 10:
            return f"Could not match designation '{designation}' to any column (best match: {best_col}, score: {best_score})."
        
    else:
        best_col = designation_cols[0]

    row = df[df.iloc[:, 0] == country]
    if row.empty:
        return f"Country/Region '{country}' not found."

    contact = row[best_col].values[0]
    return contact if pd.notna(contact) else "No contact found for that designation and region."

# --------------------------
# MAIN UI
# --------------------------

# Page header
st.markdown('<div class="hero-title">Decision Maker Discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Find the right sales contact for any country and designation instantly</div>', unsafe_allow_html=True)

st.markdown("---")
# Create two columns for layout
col1, col2 = st.columns([1, 3], gap="large", border=True)

with col1:
    st.markdown("### Upload Contact Database")
    uploaded_file = st.file_uploader(
        "Upload Excel file with contact information",
        type=["xlsx", "xls"],
        help="Upload your ManpowerGroup Sales Support Country Contact file or similar contact database"
    )
    
    if uploaded_file:
        st.markdown('<div class="success-message">File uploaded successfully! You can now start asking questions.</div>', unsafe_allow_html=True)
        
        # Show file info
        with st.expander("File Information"):
            file_info = f"""
            **Filename:** {uploaded_file.name}  
            **Size:** {uploaded_file.size:,} bytes  
            **Type:** {uploaded_file.type}
            """
            st.markdown(file_info)

with col2:
    if uploaded_file:
        try:
            df = load_excel_with_dynamic_header(uploaded_file)
            
            st.markdown("### Contact Lookup Assistant")
            st.markdown("Ask questions like: *'Who is the Main PoC in Germany?'* or *'Find the sales contact for Japan'*")
            
            # Initialize session state
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "chat_active" not in st.session_state:
                st.session_state.chat_active = True

            # Show chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        # Style assistant responses based on content
                        if "not found" in msg["content"].lower() or "could not" in msg["content"].lower():
                            st.markdown(f'<div class="error-message">{msg["content"]}</div>', unsafe_allow_html=True)
                        elif "no contact found" in msg["content"].lower():
                            st.markdown(f'<div class="info-message">{msg["content"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="success-message">**Contact Found:** {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(msg["content"])

            if st.session_state.chat_active:
                # Chat input
                user_input = st.chat_input("Ask about contacts by country and designation...")

                if user_input:
                    # Append user message
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    # Process the query
                    with st.spinner("Processing your request..."):
                        try:
                            # Extract info
                            country, designation = extract_country_designation(user_input)
                            if not country or not designation:
                                reply = "Could not understand the country or designation. Please try again being more specific."
                            else:
                                reply = get_contact_info(df, country, designation)
                        except Exception as e:
                            reply = f"Error processing request: {str(e)}"

                    # Append and display assistant message
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        if "not found" in reply.lower() or "could not" in reply.lower():
                            st.markdown(f'<div class="error-message">{reply}</div>', unsafe_allow_html=True)
                        elif "no contact found" in reply.lower():
                            st.markdown(f'<div class="info-message">{reply}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="success-message">**Contact Found:** {reply}</div>', unsafe_allow_html=True)
                st.markdown("---")
                # Control buttons
                col_btn1, col_btn2 = st.columns([1, 1], gap="large")
                with col_btn1:
                    if st.button("Clear Chat History", type="secondary"):
                        st.session_state.messages = []
                        st.rerun()
                        
                with col_btn2:
                    if st.button("End Chat Session", type="primary"):
                        st.session_state.chat_active = False
                        st.rerun()

            else:
                st.markdown('<div class="info-message">Chat session ended. Click "Start New Session" to restart.</div>', unsafe_allow_html=True)
                if st.button("Start New Session", type="primary"):
                    st.session_state.chat_active = True
                    st.session_state.messages = []
                    st.rerun()

        except Exception as e:
            st.markdown(f'<div class="error-message">Error loading file: {str(e)}</div>', unsafe_allow_html=True)
    else:
        st.markdown("### Contact Lookup Assistant")
        st.markdown('<div class="info-message">Please upload an Excel file to start using the contact lookup service.</div>', unsafe_allow_html=True)
        
        # Show example queries
        st.markdown("#### Example Queries")
        example_queries = [
            "Who is the Main PoC in Germany?",
            "Find the sales contact for Japan",
            "What is the contact for recruitment in UK?",
            "Who handles operations in Middle East?",
            "Get the main contact for Australia"
        ]
        
        for query in example_queries:
            st.markdown(f"• *{query}*")

# Footer information
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.markdown("#### How it works")
    st.markdown("""
    1. Upload your contact database Excel file
    2. Ask natural language questions about contacts
    3. Get instant responses with contact information
    """)

with col_info2:
    st.markdown("#### Supported formats")
    st.markdown("""
    - Excel files (.xlsx, .xls)
    - Automatic header detection
    - Country and designation mapping
    """)

with col_info3:
    st.markdown("#### AI Features")
    st.markdown("""
    - Natural language understanding
    - Smart country normalization
    - Fuzzy designation matching
    """)
