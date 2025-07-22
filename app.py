import streamlit as st
import pandas as pd

# Import pillar modules from new structure
from pillar.sales_acquisition import pillar_page as sales_acquisition
from pillar.recruiter_productivity import pillar_page as recruiter_productivity
from pillar.candidate_experience import pillar_page as candidate_experience
from pillar.finance_transformation import pillar_page as finance_transformation
from pillar.conversational_ai import pillar_page as conversational_ai
from pillar.technology_transformation import pillar_page as technology_transformation
from pillar.employer_branding import pillar_page as employer_branding

# Individual use case modules are now directly referenced in navigation
# No need to import them since st.Page will load them directly

# Configure the page
st.set_page_config(
    page_title="HR - CoPilot",
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

# Data structure for AI-powered recruitment & HR solutions
use_cases_data = {
    "Sales Acquisition": sales_acquisition.use_cases_data,
    "Recruiter Productivity": recruiter_productivity.use_cases_data,
    "Candidate Experience": candidate_experience.use_cases_data,
    "Finance Transformation": finance_transformation.use_cases_data,
    "Conversational AI": conversational_ai.use_cases_data,
    "Technology Transformation": technology_transformation.use_cases_data,
    "Employer Branding & Talent Marketing": employer_branding.use_cases_data
}

# Define navigation pages
def home_page():
    """Home page with hero section, pillar overview cards, and about section"""
    render_hero()
    st.divider()
    render_pillar_overview_cards()
    st.divider()
    render_about()

# Individual pillar pages - now imported from separate modules
def sales_acquisition_page():
    """Sales Acquisition pillar page"""
    sales_acquisition.render_pillar_page()

def recruiter_productivity_page():
    """Recruiter Productivity pillar page"""
    recruiter_productivity.render_pillar_page()

def candidate_experience_page():
    """Candidate Experience pillar page"""
    candidate_experience.render_pillar_page()

def finance_transformation_page():
    """Finance Transformation pillar page"""
    finance_transformation.render_pillar_page()

def conversational_ai_page():
    """Conversational AI pillar page"""
    conversational_ai.render_pillar_page()

def technology_transformation_page():
    """Technology Transformation pillar page"""
    technology_transformation.render_pillar_page()

def employer_branding_page():
    """Employer Branding & Talent Marketing pillar page"""
    employer_branding.render_pillar_page()

# Pillar descriptions for overview cards
pillar_descriptions = {
    "Sales Acquisition": "Transform your client acquisition with AI-powered tools that identify decision makers, create personalized presentations, and provide real-time market intelligence.",
    "Recruiter Productivity": "Supercharge your recruitment workflow with intelligent automation for job descriptions, candidate discovery, email outreach, and advanced search optimization.",
    "Candidate Experience": "Enhance every touchpoint of the candidate journey with personalized job matching, career development guidance, and AI-powered interview preparation.",
    "Finance Transformation": "Streamline recruitment finance operations with automated invoice processing, multi-language translation, and intelligent expense management systems.",
    "Conversational AI": "Enable natural voice interactions for candidate screening, support, and engagement through advanced AI conversation technology.",
    "Technology Transformation": "Build custom recruitment solutions with AI-powered code generation and advanced prompt engineering for your specific business needs.",
    "Employer Branding & Talent Marketing": "Create compelling employer brand content and SEO-optimized job postings that attract top talent and enhance your market presence."
}

# Pillar Overview Cards for Home Page
def render_pillar_overview_cards():
    """Render overview cards for all pillars on home page"""
    st.markdown('<div class="arsenal-title">AI-Powered Recruitment Arsenal</div>', unsafe_allow_html=True)
    st.markdown('<div class="arsenal-subtitle">Every tool you need to dominate talent acquisition and place candidates faster than ever</div>', unsafe_allow_html=True)
    st.write("")
    
    # Create pillar overview cards
    pillar_areas = list(use_cases_data.keys())
    
    # Display pillar areas in rows of 2
    for i in range(0, len(pillar_areas), 2):
        cols = st.columns(2, gap="small")
        for j, area in enumerate(pillar_areas[i:i+2]):
            if j < len(cols):
                with cols[j]:
                    with st.container(border=True):
                        st.markdown(f"### {area}")
                        
                        tool_count = len(use_cases_data[area])
                        #st.markdown(f"AI Tools Available: {tool_count}")
                        st.markdown(f'<div class="pillar-description">{pillar_descriptions.get(area, "Comprehensive AI tools for this business area.")}</div>', unsafe_allow_html=True)
                        #st.markdown(f"*Click '{area}' in the navigation to explore these tools*")

# Individual use case page functions are no longer needed 
# since we're directly pointing to the use case files in navigation

# Navigation setup with grouped pages and dropdowns
def setup_navigation():
    pages = {
        "HR Copilot": [
            st.Page(home_page, title="Home")
        ],
        "Sales": [
            st.Page(sales_acquisition_page, title="Overview"),
            st.Page("pillar/sales_acquisition/smart_document_intelligence/use_case.py", title="Smart Document Intelligence", url_path="smart_document_intelligence"),
            st.Page("pillar/sales_acquisition/personality_driven_presentations/use_case.py", title="Personality Driven Sales Pitch", url_path="personality_driven_presentations"),
            st.Page("pillar/sales_acquisition/industry_intelligence_dashboard/use_case.py", title="Industry Intelligence News", url_path="industry_intelligence_dashboard"),
            st.Page("pillar/sales_acquisition/decision_maker_discovery/use_case.py", title="Decision Maker Discovery", url_path="decision_maker_discovery"),
            st.Page("pillar/sales_acquisition/crm_optimization/use_case.py", title="CRM Data Optimization", url_path="crm_optimization"),
            st.Page("pillar/sales_acquisition/market_intelligence/use_case.py", title="Market Intelligence Reports", url_path="market_intelligence")
        ],
        "Recruiter Productivity": [
            st.Page(recruiter_productivity_page, title="Overview"),
            st.Page("pillar/recruiter_productivity/ai_job_description/use_case.py", title="AI Job Description Generator", url_path="ai_job_description"),
            st.Page("pillar/recruiter_productivity/candidate_discovery/use_case.py", title="Smart Candidate Discovery", url_path="candidate_discovery"),
            st.Page("pillar/recruiter_productivity/email_automation/use_case.py", title="Email Automation Suite", url_path="email_automation"),
            st.Page("pillar/recruiter_productivity/boolean_search/use_case.py", title="Boolean Search Optimizer", url_path="boolean_search")
        ],
        "Talent Intimacy": [
            st.Page(candidate_experience_page, title="Overview"),
            st.Page("pillar/candidate_experience/job_matching/use_case.py", title="Job Matching Engine", url_path="job_matching"),
            st.Page("pillar/candidate_experience/career_advisor/use_case.py", title="Career Development Advisor", url_path="career_advisor"),
            st.Page("pillar/candidate_experience/interview_coach/use_case.py", title="Interview Preparation Coach", url_path="interview_coach"),
            st.Page("pillar/candidate_experience/learning_recommender/use_case.py", title="Learning Recommender", url_path="learning_recommender"),
            st.Page("pillar/candidate_experience/job_compatibility/use_case.py", title="Job Compatibility Assessment", url_path="job_compatibility")
        ],
        "Finance Transformation": [
            st.Page(finance_transformation_page, title="Overview"),
            st.Page("pillar/finance_transformation/invoice_processing/use_case.py", title="Invoice Processing", url_path="invoice_processing"),
            st.Page("pillar/finance_transformation/invoice_translation/use_case.py", title="Invoice Translation", url_path="invoice_translation"),
            st.Page("pillar/finance_transformation/data_extraction/use_case.py", title="Financial Data Extraction", url_path="data_extraction"),
            st.Page("pillar/finance_transformation/po_matching/use_case.py", title="Purchase Order Matching", url_path="po_matching")
        ],
        "Conversational AI": [
            st.Page(conversational_ai_page, title="Overview"),
            st.Page("pillar/conversational_ai/voice_interaction/use_case.py", title="Voice-Enabled Interaction", url_path="voice_interaction")
        ],
        "Technology Transformation": [
            st.Page(technology_transformation_page, title="Overview"),
            st.Page("pillar/technology_transformation/tool_development/use_case.py", title="InHouse Copilot", url_path="tool_development"),
            st.Page("pillar/technology_transformation/prompt_engineering/use_case.py", title="Prompt Perfect", url_path="prompt_engineering")
        ],
        "Marketing": [
            st.Page(employer_branding_page, title="Overview"),
            st.Page("pillar/employer_branding/content_creation/use_case.py", title="Content Creation Engine", url_path="content_creation"),
            st.Page("pillar/employer_branding/seo_job_posting/use_case.py", title="SEO Job Posting Generator", url_path="seo_job_posting")
        ]
    }
    
    pg = st.navigation(pages, position="top")
    return pg

# Enhanced Hero Section
def render_hero():
    # Main Hero Content
    with st.container():
        st.markdown('<div class="hero-title">AI that thinks like your Top Recruiter</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Source faster. Place better. Scale infinitely.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-description">
        Transform your talent acquisition with AI that understands the recruiting process. 
        From onboarding clients with targeted sales to candidate sourcing and transforming across finance & technology, our Recruitment HR-AI works 24x7 to supercharge your hiring.
        </div>
        """, unsafe_allow_html=True)



    # Call to Action Section
    st.markdown("""
    <div class="cta-section">
        <div class="cta-title">Ready to 10x your recruitment game?</div>
        <div class="cta-subtitle">Join elite recruiters who place top talent while their competitors are still searching</div>
    </div>
    """, unsafe_allow_html=True)





# About Section
def render_about():
    st.markdown('<div class="arsenal-title">About Our Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="arsenal-subtitle">Meet the experts behind AI-powered recruitment innovation</div>', unsafe_allow_html=True)
    st.write("")
    
    # Team introduction
    col1, col2 = st.columns([2, 1], gap="small", border=False)
    
    with col1:
        st.markdown('''
        <div class="hero-subtitle">
            <h3 style="margin-bottom: 1rem; color: #374151; font-size: 1.3rem;">Our Mission</h3>
            <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(116, 129, 149, 0.3), transparent); margin: 0 0 1rem 0;">
            <div style="font-size: 0.9rem; line-height: 1.4;">
                <p style="margin-bottom: 0.7rem;">
                We're a team of recruitment industry veterans, AI researchers, and technology innovators united by a single vision: 
                <strong>transforming how talent acquisition works in the modern world.</strong>
                </p>
                <p style="margin-bottom: 0;">
                After years of witnessing the challenges faced by recruiters—from time-consuming manual processes to the struggle 
                of finding the right candidates in an increasingly competitive market—we decided to build something revolutionary.
                </p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="hero-subtitle">
            <h3 style="margin-bottom: 1rem; color: #374151; font-size: 1.3rem;">By the Numbers</h3>
            <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(116, 129, 149, 0.3), transparent); margin: 0 0 1rem 0;">
            <ul style="list-style: none; padding: 0; text-align: left; font-size: 0.9rem; line-height: 1.4;">
                <li style="margin-bottom: 0.7rem;"><strong>5+ Years</strong> of product development in recruitment</li>
                <li style="margin-bottom: 0.7rem;"><strong>10+ Years</strong> of AI research in recruitment</li>
                <li style="margin-bottom: 0.7rem;"><strong>25+ AI Tools</strong> deployed across 7 business areas</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    st.write("")
    
    # Team highlights
    st.markdown('<h3 style="text-align: center; color: #374151; font-family: var(--header-font); font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem;">Our Expertise</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown('''
        <div class="hero-subtitle">
            <h4 style="margin-bottom: 1rem; color: #374151; font-size: 1.3rem;">Industry Veterans</h4>
            <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(116, 129, 149, 0.3), transparent); margin: 0 0 1rem 0;">
            <ul style="list-style: none; padding: 0; text-align: center; font-size: 0.9rem; line-height: 1.4;">
                <li style="margin-bottom: 0.7rem;">Yashwant Sai</li>
                <li style="margin-bottom: 0.7rem;">Prerit Saxena</li>
                <li style="margin-bottom: 0.7rem;">Praneeth Paikray</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="hero-subtitle">
            <h4 style="margin-bottom: 1rem; color: #374151; font-size: 1.3rem;">AI Researchers</h4>
            <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(116, 129, 149, 0.3), transparent); margin: 0 0 1rem 0;">
            <ul style="list-style: none; padding: 0; text-align: center; font-size: 0.9rem; line-height: 1.4;">
                <li style="margin-bottom: 0.7rem;">Anurag Dubey</li>
                <li style="margin-bottom: 0.7rem;">Samaroha Chatterjee</li>
                <li style="margin-bottom: 0.7rem;">...</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div class="hero-subtitle">
            <h4 style="margin-bottom: 1rem; color: #374151; font-size: 1.3rem;">Product Innovators</h4>
            <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(116, 129, 149, 0.3), transparent); margin: 0 0 1rem 0;">
            <ul style="list-style: none; padding: 0; text-align: center; font-size: 0.9rem; line-height: 1.4;">
                <li style="margin-bottom: 0.7rem;">Pratyush Mishra</li>
                <li style="margin-bottom: 0.7rem;">Vignesh Manjula</li>
                <li style="margin-bottom: 0.7rem;">Harsh Saxena</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)

# Main App Logic
def main():
    # Set up navigation with proper dropdown functionality
    selected_page = setup_navigation()
    
    # Run the selected page (navigation handles everything now)
    selected_page.run()

if __name__ == "__main__":
    main()


