import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import mimetypes
import io
from pypdf import PdfReader
import json
from datetime import datetime
from typing import List, Dict, Optional
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "your_serpapi_key_here")

# --- OpenRouter Client Setup ---
def get_openrouter_client():
    """Get configured OpenRouter client using OpenAI package"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

def call_moonshot_ai(prompt: str, max_tokens: int = 4000) -> str:
    """Make API call to Moonshot AI via OpenRouter using OpenAI client"""
    try:
        client = get_openrouter_client()
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://hr-copilot.com",  # Optional
                "X-Title": "Industry Intelligence Dashboard",  # Optional
            },
            model="moonshotai/kimi-k2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI API Error: {str(e)}")
        return "Error: Unable to process request with AI"

def search_serpapi(query: str, num_results: int = 8) -> List[str]:
    """Search using SerpAPI and return URLs"""
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "hl": "en",
            "gl": "us"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("organic_results", [])
            return [r.get("link") for r in results[:num_results] if r.get("link")]
        else:
            st.warning(f"Search API returned status {response.status_code}")
            return []
            
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file"""
    if url.endswith(".pdf"):
        return True
    try:
        mime_type, _ = mimetypes.guess_type(url)
        return mime_type == "application/pdf"
    except:
        return False

def fetch_pdf_text(url: str) -> str:
    """Extract text from PDF URL"""
    try:
        response = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        if response.status_code == 200:
            with io.BytesIO(response.content) as pdf_io:
                reader = PdfReader(pdf_io)
                text = ""
                # Read first 5 pages to avoid overwhelming the AI
                for page in reader.pages[:5]:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text[:6000]  # Limit text length
        else:
            return f"[PDF fetch failed - Status: {response.status_code}]"
            
    except Exception as e:
        return f"[PDF processing error: {str(e)}]"

def fetch_html_text(url: str) -> str:
    """Extract text from HTML URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, timeout=15, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text from paragraphs and divs
            text_elements = soup.find_all(['p', 'div', 'article', 'section'], limit=30)
            text = '\n'.join([elem.get_text(strip=True) for elem in text_elements if elem.get_text(strip=True)])
            
            return text[:6000]  # Limit text length
        else:
            return f"[HTML fetch failed - Status: {response.status_code}]"
            
    except Exception as e:
        return f"[HTML processing error: {str(e)}]"

def fetch_content(url: str) -> str:
    """Fetch and extract content from URL"""
    if is_pdf_url(url):
        return fetch_pdf_text(url)
    else:
        return fetch_html_text(url)

def generate_intelligence_report(content: str, company: str, location: str, source_urls: List[str], years: List[str]) -> str:
    """Generate comprehensive intelligence report using Moonshot AI"""
    
    # Create source reference mapping
    source_refs = "\n".join([f"Source {i+1}: {url}" for i, url in enumerate(source_urls)])
    years_filter = ", ".join(years)
    
    prompt = f"""
    As a senior business intelligence analyst specializing in recruitment and staffing services, create a comprehensive executive briefing about {company} for a sales team preparing to pitch staffing solutions.

    **CRITICAL FILTERING REQUIREMENT:**
    ONLY include information from {years_filter}. Completely ignore and exclude any news, data, or information from years outside of {years_filter}. If information doesn't have a clear date from {years_filter}, do not include it in the analysis.

    Format your response as clean markdown with proper headers, bullet points, and structure. When referencing specific information, cite the source number and mention the year (e.g., "according to Source 2 (2025)").

    ## Executive Summary
    Provide a 3-4 sentence executive summary of key findings and opportunities from {years_filter} only.

    ## Source Overview
    - Summarize the key information found across sources from {years_filter}
    - Note the reliability and recency of the data
    - List sources with their reference numbers and URLs
    - Clearly state the time period covered: {years_filter}

    ## Company Intelligence ({years_filter})
    - Current business focus and strategic initiatives from {years_filter}
    - Recent expansions, acquisitions, or major announcements from {years_filter}
    - Key business units and their growth trajectories in {years_filter}
    - Geographic presence and expansion plans announced in {years_filter}

    ## Workforce & Hiring Insights ({years_filter})
    - Current hiring trends and volume indicators from {years_filter}
    - Specific roles or departments showing growth in {years_filter}
    - Remote work policies and workplace changes in {years_filter}
    - Diversity, equity & inclusion initiatives from {years_filter}
    - Any mentioned staffing challenges or talent gaps in {years_filter}

    ## Business Expansion Signals ({years_filter})
    - New office openings or relocations announced in {years_filter}
    - Product launches requiring additional workforce from {years_filter}
    - Market expansion plans from {years_filter}
    - Technology investments indicating growth in {years_filter}
    - Partnership announcements with hiring implications from {years_filter}

    ## Staffing Opportunity Assessment
    - High-potential areas for staffing partnerships based on {years_filter} data
    - Specific recruitment pain points we could solve based on {years_filter} insights
    - Timing considerations for outreach based on {years_filter} trends
    - Decision-maker insights from {years_filter} sources

    ## Competitive Landscape ({years_filter})
    - Industry challenges affecting hiring in {years_filter}
    - Market position and competitive pressures in {years_filter}
    - Regulatory changes impacting workforce needs in {years_filter}

    ## Recommended Sales Approach
    - Key value propositions to emphasize based on {years_filter} findings
    - Timing recommendations for outreach
    - Specific solutions to highlight based on {years_filter} needs
    - Questions to ask in initial conversations

    **CONTENT TO ANALYZE:**
    Location Focus: {location}
    Time Period: {years_filter} ONLY
    
    **SOURCE REFERENCES:**
    {source_refs}
    
    **CONTENT:**
    {content[:5000]}
    
    **REQUIREMENTS:**
    - Use proper markdown formatting
    - Include source citations with years (e.g., "Source 1 (2025)", "Source 2 (2024)")
    - STRICTLY filter content to {years_filter} only
    - Prioritize actionable insights from the specified time period
    - Include confidence levels where appropriate
    - Keep sections focused and scan-friendly
    - Exclude any information that cannot be confirmed to be from {years_filter}
    """
    
    return call_moonshot_ai(prompt, max_tokens=4000)

# --- Streamlit UI ---

def render_header():
    """Render the page header with consistent styling"""
    st.markdown('<div class="hero-title">Industry Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-description">AI-powered company research and workforce intelligence for strategic sales positioning</div>', 
        unsafe_allow_html=True
    )

def render_search_form():
    """Render the search input form"""
    st.markdown('<div class="arsenal-title">Research Configuration</div>', unsafe_allow_html=True)
    
    # Add custom CSS for form styling
    st.markdown("""
    <style>
    /* Progress bar styling - matching coral theme */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #d4896a 0%, #c67651 100%) !important;
        border-radius: 6px !important;
    }
    .stProgress > div > div > div {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(212, 137, 106, 0.3) !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 8px rgba(212, 137, 106, 0.2) !important;
    }
    
    /* Radio button styling - matching coral/orange theme */
    .stRadio > div {
        flex-direction: row;
        gap: 0.75rem;
        justify-content: flex-start;
        flex-wrap: wrap;
    }
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(212, 137, 106, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.25rem !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        margin: 0.25rem !important;
        white-space: nowrap !important;
        box-shadow: 0 4px 15px rgba(212, 137, 106, 0.2) !important;
    }
    .stRadio > div > label:hover {
        background: linear-gradient(135deg, rgba(212, 137, 106, 0.15) 0%, rgba(198, 118, 81, 0.15) 100%) !important;
        border: 1px solid rgba(212, 137, 106, 0.6) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(212, 137, 106, 0.4) !important;
    }
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #d4896a 0%, #c67651 100%) !important;
        border: 1px solid #c67651 !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(212, 137, 106, 0.4) !important;
    }
    .stRadio > div > label > div[data-testid="stMarkdownContainer"] > p {
        color: #374151 !important;
        font-weight: 600 !important;
        margin: 0 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .stRadio > div > label[data-checked="true"] > div[data-testid="stMarkdownContainer"] > p {
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Multiselect styling with coral theme */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 137, 106, 0.3);
        border-radius: 8px;
    }
    .stMultiSelect > div > div:hover {
        border: 1px solid rgba(212, 137, 106, 0.5);
    }
    .stMultiSelect span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #d4896a 0%, #c67651 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 137, 106, 0.3);
        border-radius: 8px;
    }
    .stTextInput > div > div > input:focus {
        border: 1px solid rgba(212, 137, 106, 0.6);
        box-shadow: 0 0 0 0.2rem rgba(212, 137, 106, 0.25);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 137, 106, 0.3);
        border-radius: 8px;
    }
    .stSelectbox > div > div:hover {
        border: 1px solid rgba(212, 137, 106, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Row 1: Company Name, Geographic Focus, and Time Period
    col1, col2, col3 = st.columns(3)
    
    with col1:
        company = st.text_input(
            "Company Name",
            placeholder="Enter company name (e.g., Microsoft, Tesla, etc.)",
            help="Enter the full company name for best results"
        )
    
    with col2:
        location = st.selectbox(
            "Geographic Focus",
            ["Global", "United States", "Canada", "United Kingdom", "Germany", "India", "Singapore", "Australia"],
            help="Select geographic region for targeted insights"
        )
    
    with col3:
        years = st.multiselect(
            "Time Period",
            ["2025", "2024", "2023", "2022", "2021", "2020"],
            default=["2025"],
            help="Select one or multiple years for research focus"
        )
    
    # Row 2: Research Depth (centered)
    st.write("")  # Add some spacing
    
    # Center the research depth options
    col_spacer1, col_center, col_spacer2 = st.columns([4, 1, 1])
    
    with col_spacer1:
        search_scope = st.radio(
            "Research Depth",
            ["Standard (5)", "Comprehensive (8)", "Deep Dive (10)"],
            horizontal=False
        )
    
    # Row 3: Generate Button (spans all columns)
    st.write("")  # Add some spacing
    
    # Map search scope to number of results
    num_results = {"Standard (5)": 5, "Comprehensive (8)": 8, "Deep Dive (10)": 10}[search_scope]
    
    # Format years for search query
    if not years:
        years = ["2024"]  # Default fallback
    
    return company, location, num_results, years

def render_search_progress(query: str, links: List[str]):
    """Render search progress and found links"""
    st.markdown('<div class="arsenal-title">Research Progress</div>', unsafe_allow_html=True)
    
    st.success(f"**Search Query:** {query}")
    st.info(f"**Found {len(links)} relevant sources** - Processing content...")
    
    if links:
        with st.expander("View Source URLs", expanded=False):
            for i, link in enumerate(links, 1):
                st.markdown(f"{i}. {link}")

def render_intelligence_report(report: str):
    """Render the final intelligence report with clean styling"""
    st.markdown('<div class="arsenal-title">Executive Intelligence Briefing</div>', unsafe_allow_html=True)
    
    # Add report metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Report Generated", datetime.now().strftime("%Y-%m-%d"))
    with col2:
        st.metric("Analysis Type", "AI-Powered")
    with col3:
        st.metric("Confidence Level", "High")
    
    # Display the report with clean markdown formatting
    st.markdown(report)

def render_error_state(error_msg: str):
    """Render error state with helpful information"""
    st.error(f"Error: {error_msg}")
    
    with st.expander("Troubleshooting Tips"):
        st.markdown("""
        **Common issues and solutions:**
        
        1. **API Key Issues:** Ensure your OpenRouter and SerpAPI keys are properly configured
        2. **Company Not Found:** Try variations of the company name or check spelling
        3. **Network Issues:** Check your internet connection and try again
        4. **Rate Limits:** Wait a few moments before trying another search
        
        **Need Help?** Contact your system administrator or check the documentation.
        """)

def main():
    """Main application function"""
    # Render header
    render_header()
    
    # Check API configuration
    if OPENROUTER_API_KEY == "your_openrouter_api_key_here" or SERPAPI_KEY == "your_serpapi_key_here":
        st.warning("**Configuration Required:** Please set your OPENROUTER_API_KEY and SERPAPI_KEY environment variables.")
        st.stop()
    
    # Render search form
    company, location, num_results, years = render_search_form()
    
    # Row 3: Generate Button (spans both columns)
    if st.button("Generate Intelligence Report", type="primary", use_container_width=True):
        if not company.strip():
            st.warning("Please enter a company name to begin research.")
            return
        
        # Construct search query with strict year filtering
        if len(years) > 1:
            years_filter = " OR ".join([f"after:{year}-01-01 before:{year}-12-31" for year in years])
            years_str = " OR ".join(years)
        else:
            year = years[0]
            years_filter = f"after:{year}-01-01 before:{year}-12-31"
            years_str = year
        
        if location == "Global":
            search_query = f"{company} workforce trends hiring expansion business news {years_str} ({years_filter})"
        else:
            search_query = f"{company} workforce trends hiring expansion business news {location} {years_str} ({years_filter})"
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Search for sources
            status_text.text("Searching for relevant sources...")
            progress_bar.progress(20)
            
            search_links = search_serpapi(search_query, num_results)
            
            if not search_links:
                render_error_state("No sources found. Try a different company name or check your internet connection.")
                return
            
            # Step 2: Display search progress
            render_search_progress(search_query, search_links)
            progress_bar.progress(40)
            
            # Step 3: Fetch and analyze content
            status_text.text("Analyzing content from sources...")
            
            all_content = []
            for i, link in enumerate(search_links):
                progress_bar.progress(40 + (i * 40 // len(search_links)))
                status_text.text(f"Processing source {i+1} of {len(search_links)}...")
                
                content = fetch_content(link)
                if content and not content.startswith("[") and len(content) > 100:
                    all_content.append(f"Source {i+1} ({link}):\n{content}\n---\n")
                
                time.sleep(0.5)  # Rate limiting
            
            # Step 4: Generate intelligence report
            progress_bar.progress(80)
            status_text.text("Generating AI-powered intelligence report...")
            
            if all_content:
                combined_content = "\n".join(all_content)
                intelligence_report = generate_intelligence_report(combined_content, company, location, search_links, years)
                
                progress_bar.progress(100)
                status_text.text("Report generated successfully!")
                
                # Clear progress indicators
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                # Render final report
                render_intelligence_report(intelligence_report)
                
            else:
                render_error_state("Unable to extract meaningful content from sources. Try a different search term.")
                
        except Exception as e:
            render_error_state(f"An unexpected error occurred: {str(e)}")
        
        finally:
            # Clean up progress indicators
            try:
                progress_bar.empty()
                status_text.empty()
            except:
                pass

if __name__ == "__main__":
    main()
