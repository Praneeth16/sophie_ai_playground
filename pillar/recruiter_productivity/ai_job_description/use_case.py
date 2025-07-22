from dotenv import load_dotenv
import streamlit as st
import os
from openai import OpenAI
from duckduckgo_search import DDGS
import pandas as pd


load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Job Description Generator",
    page_icon="artifacts/mpg.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed",
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

# --- API Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

# Initialize OpenAI client for OpenRouter
def get_openrouter_client():
    """Initialize OpenRouter client using OpenAI SDK"""
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
        return None
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

# --- Helper Functions ---

def call_openrouter_api(model, prompt, temperature=0.7, max_tokens=2000):
    """
    Makes a call to the OpenRouter API using OpenAI client.
    """
    client = get_openrouter_client()
    if client is None:
        st.error("FATAL: OpenRouter API key is not set. Please set the OPENROUTER_API_KEY environment variable.")
        return None

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://hr-copilot.streamlit.app",
                "X-Title": "HR CoPilot - AI Job Description Generator",
            },
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"API Request Error: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour since company info doesn't change frequently
def search_company_info(company_name):
    """
    Uses DuckDuckGo to find relevant information about a company.
    Cached for 1 hour since company information doesn't change frequently.
    """
    try:
        with st.spinner(f"Researching company: {company_name}..."):
            with DDGS() as ddgs:
                # Enhanced search query for better company intelligence
                search_queries = [
                    f"{company_name} company culture mission values about",
                    f"{company_name} company benefits perks workplace",
                    f"{company_name} recent news achievements awards"
                ]
                
                all_results = []
                for query in search_queries:
                    results = list(ddgs.text(query, max_results=3))
                    all_results.extend(results)
                
                if not all_results:
                    st.warning(f"No search results found for '{company_name}'. Job description will use generic company information.")
                    return "No specific company information found."
                
                # Combine and clean results
                combined_info = " ".join([r['body'] for r in all_results])
                # Limit to reasonable length
                return combined_info[:2000] if len(combined_info) > 2000 else combined_info
                
    except Exception as e:
        st.error(f"Company research failed: {e}")
        return "Company research unavailable."

# --- Enhanced Agent Functions ---

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def experience_extraction_agent(model, job_title, recruiter_summary):
    """
    Agent 1: Extract experience requirements and analyze role level.
    Cached for 30 minutes since job analysis is consistent for same inputs.
    """
    prompt = f"""
    You are an expert in analyzing job requirements and extracting key experience criteria.
    
    **Task:** Analyze the job title "{job_title}" and description to extract experience requirements.
    
    **Analysis Areas:**
    1. Years of experience required (extract from description or infer from title)
    2. Seniority level (entry, mid, senior, lead, principal)
    3. Key technologies and skills mentioned
    4. Industry context and specialization
    
    **Job Title:** {job_title}
    **Job Description:** {recruiter_summary}
    
    **Output Format:**
    - Experience Level: [X years / Entry / Mid / Senior / Lead]
    - Seniority: [Level description]
    - Key Skills: [List main technologies/skills]
    - Specialization: [Domain/industry focus]
    
    Be concise and focus on extracting concrete requirements.
    """
    
    with st.spinner("Analyzing experience requirements..."):
        return call_openrouter_api(model, prompt, temperature=0.1)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def market_research_agent(model, job_title, company_info):
    """
    Agent 2: Market research and competitive analysis.
    Cached for 30 minutes since market research is consistent for same inputs.
    """
    prompt = f"""
    You are a senior talent market researcher specializing in recruitment intelligence.
    
    **Mission:** Provide market context and competitive insights for the role "{job_title}".
    
    **Available Context:**
    - Company Information: {company_info}
    
    **Research Areas:**
    1. Current market demand for this role
    2. Key skills and qualifications in demand
    3. Typical career progression paths
    4. Industry-specific requirements
    5. Competitive landscape considerations
    
    **Deliverable:** A comprehensive market overview (200-300 words) that includes:
    - Market demand indicators
    - Critical skills and qualifications
    - Career advancement opportunities
    - Industry context and trends
    
    Focus on actionable insights that will help attract top talent.
    """
    
    with st.spinner("Conducting market research..."):
        return call_openrouter_api(model, prompt, temperature=0.3)

@st.cache_data(ttl=900)  # Cache for 15 minutes - shorter since this includes dynamic elements
def job_description_architect(model, job_title, experience_analysis, company_name, company_info, recruiter_summary, market_research, salary_info, remote_policy, location):
    """
    Agent 3: Master job description architect.
    Cached for 15 minutes since job descriptions may need customization.
    """
    prompt = f"""
    You are a world-class talent acquisition specialist and job description architect.
    
    **Mission:** Create an exceptional job description that attracts A-players and drives applications.
    
    **Context:**
    - **Position:** {job_title}
    - **Experience Analysis:** {experience_analysis}
    - **Company:** {company_name}
    - **Company Intelligence:** {company_info}
    - **Hiring Manager Brief:** {recruiter_summary}
    - **Market Research:** {market_research}
    - **Salary Information:** {salary_info if salary_info else 'Not specified'}
    - **Work Policy:** {remote_policy if remote_policy else 'Not specified'}
    - **Location:** {location if location else 'Not specified'}
    
    **CRITICAL FORMATTING REQUIREMENTS:**
    - Output ONLY clean, valid Markdown
    - Use standard heading levels (## for sections, ### for subsections)
    - Use bullet points with - or * for lists
    - Do NOT use any special characters or formatting that might break rendering
    - Do NOT include any explanations before or after the job description
    - Start directly with the job description content
    
    **Architecture Framework:**
    
    ## {job_title}
    
    ### About {company_name}
    [Compelling company description based on research]
    
    ### The Role
    [Clear scope and expectations]
    
    ### What You'll Do
    - [Action-oriented responsibility 1]
    - [Action-oriented responsibility 2]
    - [Action-oriented responsibility 3]
    
    ### What You'll Bring
    - [Must-have qualification 1]
    - [Must-have qualification 2]
    - [Nice-to-have qualification 3]
    
    ### Compensation & Benefits
    [Include if salary_info provided]
    
    ### Work Arrangement
    [Include if remote_policy provided]
    
    ### Location
    [Include if location provided]
    
    ### Why Join Us?
    [Compelling close and call to action]
    
    **Quality Standards:**
    - Use power verbs and specific outcomes
    - Include metrics where appropriate
    - Balance technical requirements with growth opportunities
    - Emphasize impact and career development
    - Maintain professional yet engaging tone
    - Ensure gender-neutral language
    - Include salary, work policy, and location information naturally if provided
    
    **IMPORTANT:** Return ONLY the formatted job description in markdown. No extra text, explanations, or formatting notes.
    """
    
    with st.spinner("Architecting job description..."):
        return call_openrouter_api(model, prompt, temperature=0.4)

@st.cache_data(ttl=900)  # Cache for 15 minutes
def optimization_agent(model, job_description_draft, job_title):
    """
    Agent 4: Final optimization for ATS, SEO, and candidate appeal.
    Cached for 15 minutes since optimization logic is consistent.
    """
    prompt = f"""
    You are an expert in recruitment optimization, specializing in ATS compatibility, SEO, and candidate psychology.
    
    **Mission:** Transform this job description into a high-performing, optimized posting.
    
    **Draft to Optimize:**
    {job_description_draft}
    
    **CRITICAL FORMATTING REQUIREMENTS:**
    - Return ONLY clean, valid Markdown
    - Maintain the existing structure and formatting
    - Do NOT add any explanations, notes, or extra text
    - Ensure all markdown syntax is correct
    - Start directly with the optimized job description
    
    **Optimization Checklist:**
    
    **ATS Optimization:**
    - Ensure keyword density for "{job_title}" and related terms
    - Include standard section headers
    - Use bullet points for scannability
    - Avoid complex formatting that breaks parsing
    
    **SEO Enhancement:**
    - Integrate relevant keywords naturally
    - Optimize for job search platforms
    - Include location and industry terms
    
    **Candidate Psychology:**
    - Lead with benefits and growth opportunities
    - Use inclusive language
    - Create urgency without pressure
    - Highlight unique value propositions
    
    **Quality Assurance:**
    - Remove any biased language
    - Ensure clarity and conciseness
    - Verify professional tone
    - Check for grammatical perfection
    - Ensure proper markdown formatting
    
    **IMPORTANT:** Return ONLY the final, optimized job description in clean markdown format. No additional text, comments, or formatting notes.
    """
    
    with st.spinner("Optimizing for maximum impact..."):
        return call_openrouter_api(model, prompt, temperature=0.2)

# --- Cache Management Utilities ---
def clear_all_caches():
    """Clear all cached data for fresh results"""
    search_company_info.clear()
    experience_extraction_agent.clear()
    market_research_agent.clear()
    job_description_architect.clear()
    optimization_agent.clear()

def get_cache_status():
    """Get cache status information"""
    cache_info = {
        "Company Research": "Cached for 1 hour",
        "Experience Analysis": "Cached for 30 minutes", 
        "Market Research": "Cached for 30 minutes",
        "Job Architecture": "Cached for 15 minutes",
        "Final Optimization": "Cached for 15 minutes"
    }
    return cache_info

def clean_markdown_content(content):
    """Clean and validate markdown content for proper Streamlit rendering"""
    if not content:
        return "No content generated."
    
    # Remove any leading/trailing whitespace
    content = content.strip()
    
    # Remove any potential AI response prefixes or suffixes
    if content.startswith("Here's") or content.startswith("Here is"):
        lines = content.split('\n')
        content = '\n'.join(lines[1:]).strip()
    
    # Remove any trailing explanations or notes
    if "Note:" in content or "Please note:" in content:
        content = content.split("Note:")[0].strip()
        content = content.split("Please note:")[0].strip()
    
    # Ensure proper line breaks for better rendering
    content = content.replace('\n\n\n', '\n\n')
    
    return content

# --- Streamlit UI ---

# --- Header Section ---
st.markdown('<div class="hero-title">AI Job Description Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Transform brief requirements into compelling job postings that attract top talent</div>', unsafe_allow_html=True)

st.markdown("---")

# --- Main Content Layout ---

with st.container(border=True):
    st.markdown('<div class="arsenal-title">Job Requirements</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-description">Provide the essential details below. Our AI agents will handle the research, writing, and optimization.</div>', unsafe_allow_html=True)
    
    # Input Form
    with st.container(border=True):
        # Basic Information
        st.markdown("### Basic Information")
        
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            job_title = st.text_input(
                "Job Title",
                placeholder="e.g., Senior Backend Engineer",
                help="Enter the exact job title as it should appear on the posting"
            )
        
        with input_col2:
            company_name = st.text_input(
                "Company Name",
                placeholder="e.g., TechCorp Inc.",
                help="Company name for research and personalization"
            )
        
        st.markdown("### Role Summary & Requirements")
        recruiter_summary = st.text_area(
            "Hiring Manager Brief",
            placeholder="Example: We need a senior backend engineer with 5+ years experience in Python and distributed systems. The role involves architecting scalable microservices, mentoring junior developers, and collaborating with product teams. Must have experience with AWS, Docker, and PostgreSQL. Remote-friendly with quarterly team meetups.",
            height=150,
            help="Describe the role, key requirements, and any specific details about the position"
        )
        
        # Advanced Options
        with st.expander("Advanced Options"):
            st.markdown("### Additional Information")
            
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                salary_info = st.text_input(
                    "Salary/Compensation (Optional)",
                    placeholder="e.g., $120,000 - $150,000 + equity + benefits",
                    help="Enter salary range, benefits, or compensation details to include"
                )
                
                remote_policy = st.text_input(
                    "Work Policy (Optional)", 
                    placeholder="e.g., Remote-first, Hybrid (3 days office), On-site required",
                    help="Describe the work arrangement - remote, hybrid, or on-site"
                )
            
            with col_adv2:
                location = st.text_input(
                    "Location (Optional)",
                    placeholder="e.g., San Francisco, CA or New York, NY",
                    help="Job location for better targeting and SEO"
                )
            
            if st.button("Clear All Cache", help="Clear cached data to force fresh API calls"):
                clear_all_caches()
                st.success("Cache cleared! Next generation will use fresh data.")
            
            
# Set the model to use
selected_model = "moonshotai/kimi-k2"

# --- Centered Generation Section ---
st.markdown("---")

# Center the generation button
col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
with col_center2:
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    generate_button = st.button("Generate Job Description", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if generate_button:
    # Input Validation
    if not all([job_title, company_name, recruiter_summary]):
        st.warning("Please fill in all required fields: Job Title, Company Name, and Hiring Manager Brief.")
    elif OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
        st.error("Please set your OPENROUTER_API_KEY environment variable before generating.")
    else:
        # Clear previous results from session state
        if 'generated_job_description' in st.session_state:
            del st.session_state.generated_job_description
        if 'job_analysis_data' in st.session_state:
            del st.session_state.job_analysis_data
            
        # Progress tracking
        progress_bar = st.progress(0, text="Initializing AI agents...")
        
        try:
            # Agent 1: Experience Extraction
            progress_bar.progress(20, text="Analyzing experience requirements...")
            experience_analysis = experience_extraction_agent(selected_model, job_title, recruiter_summary)
            if not experience_analysis:
                st.error("Experience analysis failed. Please try again.")
                st.stop()
            
            # Agent 2: Company Research
            progress_bar.progress(40, text="Researching company and market...")
            company_info = search_company_info(company_name)
            market_research = market_research_agent(selected_model, job_title, company_info)
            if not market_research:
                st.error("Market research failed. Please try again.")
                st.stop()
            
            # Agent 3: Job Description Architecture
            progress_bar.progress(70, text="Crafting job description...")
            initial_draft = job_description_architect(
                selected_model, job_title, experience_analysis, company_name, 
                company_info, recruiter_summary, market_research, salary_info, remote_policy, location
            )
            if not initial_draft:
                st.error("Job description generation failed. Please try again.")
                st.stop()
            
            # Agent 4: Final Optimization
            progress_bar.progress(90, text="Optimizing for maximum impact...")
            final_job_description = optimization_agent(selected_model, initial_draft, job_title)
            if not final_job_description:
                st.error("Optimization failed. Please try again.")
                st.stop()
            
            # Clean the markdown content for proper rendering
            final_job_description = clean_markdown_content(final_job_description)
            
            # Store results in session state for persistence
            st.session_state.generated_job_description = final_job_description
            st.session_state.job_analysis_data = {
                'experience_analysis': experience_analysis,
                'job_title': job_title,
                'generation_time': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            progress_bar.progress(100, text="Complete!")
            
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            progress_bar.empty()

# Display results if available in session state
if 'generated_job_description' in st.session_state:
    # --- Display Results ---
    st.success("Your optimized job description is ready!")
    
    # Results Layout
    with st.container(border=True):
        st.markdown("## Generated Job Description")
        st.markdown("---")
        
        final_job_description = st.session_state.generated_job_description
        job_analysis_data = st.session_state.get('job_analysis_data', {})
        
        # Check if final_job_description has content and render
        if final_job_description and len(final_job_description.strip()) > 0:
            # Try rendering with st.markdown first
            try:
                st.markdown(final_job_description, unsafe_allow_html=False)
            except Exception as e:
                st.error(f"Markdown rendering failed: {e}")
                # Fallback: Use st.write which is more forgiving
                st.write("**Fallback Display:**")
                st.write(final_job_description)
        else:
            st.error("No job description content generated. Please try again.")
        
        st.markdown("---")
        
        # Action buttons
        col_download, col_cache, col_new = st.columns(3)
        
        with col_download:
            job_title_for_file = job_analysis_data.get('job_title', 'job_description')
            st.download_button(
                label="Download Job Description",
                data=final_job_description,
                file_name=f"{job_title_for_file.replace(' ', '_').lower()}_job_description.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download the job description as a text file"
            )
        
        with col_cache:
            if st.button("Clear Cache", help="Clear cache for fresh results", key="clear_cache_results", use_container_width=True):
                clear_all_caches()
                st.success("Cache cleared!")
        
        with col_new:
            if st.button("Generate New", help="Create a new job description", key="generate_new", use_container_width=True):
                # Clear session state to start fresh
                if 'generated_job_description' in st.session_state:
                    del st.session_state.generated_job_description
                if 'job_analysis_data' in st.session_state:
                    del st.session_state.job_analysis_data
                st.rerun()