import streamlit as st
import os
import time
import base64
import tempfile
from openai import OpenAI
import pickle
import json
import re
from datetime import datetime
import pandas as pd
from typing import List, Optional, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from llama_cloud_services import LlamaParse

# OpenRouter API Configuration
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "your-openrouter-api-key")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-2.5-pro"

# LlamaCloud Configuration
LLAMACLOUD_API_KEY = st.secrets.get("LLAMACLOUD_API_KEY", "your-llamacloud-api-key")

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

# Initialize LlamaCloud client
llama_cloud = LlamaParse(api_key=LLAMACLOUD_API_KEY)

# Create sessions directory if not exists
if not os.path.exists("sessions"):
    os.makedirs("sessions")

def clean_json_string(json_str):
    """Clean JSON string by removing invalid control characters"""
    # Remove control characters except for \n, \r, \t
    json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
    
    # Replace any remaining problematic characters
    json_str = json_str.replace('\x00', '').replace('\x08', '').replace('\x0c', '')
    
    # Ensure the string is properly encoded
    json_str = json_str.encode('utf-8', errors='ignore').decode('utf-8')
    
    return json_str.strip()

def safe_json_parse(json_str):
    """Safely parse JSON string with error handling"""
    try:
        # First, clean the JSON string
        clean_str = clean_json_string(json_str)
        
        # Try to parse the cleaned JSON
        return json.loads(clean_str)
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        st.error(f"Problematic JSON string (first 200 chars): {json_str[:200]}")
        
        # Try to extract JSON from the string if it's wrapped in other text
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_part = json_match.group(0)
                clean_part = clean_json_string(json_part)
                return json.loads(clean_part)
        except:
            pass
        
        return None

# Load DISC types
@st.cache_data
def load_disc_types():
    """Load DISC types from CSV file"""
    try:
        disc_df = pd.read_csv("data/disc_types.csv")
        return disc_df
    except Exception as e:
        st.error(f"Error loading DISC types: {e}")
        return pd.DataFrame()

# Function to extract text from PDF using LlamaParse
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file using LlamaParse"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Parse the PDF using LlamaParse
        result = llama_cloud.parse(tmp_file_path)
        
        # Get text documents without page splitting
        text_documents = result.get_markdown_documents(split_by_page=False)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return text_documents.strip()
        
    except Exception as e:
        st.error(f"Error extracting text from PDF using LlamaParse: {e}")
        return ""

def extract_basic_info(text):
    """Extract basic information from LinkedIn profile text"""
    lines = text.strip().split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    
    name = "Unknown"
    position = "Not Available"
    company = "Not Available"
    
    # Simple extraction logic - can be enhanced
    for i, line in enumerate(lines):
        if i < 10:  # Look in first 10 lines
            if " at " in line and position == "Not Available":
                parts = line.split(" at ")
                if len(parts) >= 2:
                    position = parts[0].strip()
                    company = parts[1].strip()
            elif name == "Unknown" and len(line.split()) <= 4 and not any(char in line for char in ["•", "|", "-", "→"]):
                name = line.strip()
    
    return name, position, company

@st.cache_data
def analyze_linkedin_profile(profile_text: str) -> Dict[str, Any]:
    """Analyze LinkedIn profile and return structured analysis"""
    
    # Clean the profile text to remove any problematic characters
    clean_profile_text = clean_json_string(profile_text)
    
    # First LLM call: Extract basic information and personality traits
    prompt1 = f"""
    Analyze the following LinkedIn profile pdf and extract key information. Return ONLY a valid JSON object with the following structure:
    {{
        "name": "Extract the Linkedin Profile Person's name mentioned at the beginning of the profile",
        "position": "Current position or title mentioned in the profile", 
        "company": "Current company mentioned in the profile",
        "location": "Current location mentioned in the profile",
        "personality_traits": {{
            "risk_tolerance": 45,
            "trust_level": 60,
            "optimism_level": 30,
            "decision_style": 70,
            "communication_style": 40,
            "work_style": 65,
            "leadership_style": 55
        }},
        "key_insights": ["insight1", "insight2", "insight3"],
        "communication_preferences": ["pref1", "pref2", "pref3"]
    }}

    LinkedIn Profile:
    {clean_profile_text[:10_000]}

    Return only the JSON object, no additional text or explanation.
    """
    
    try:
        response1 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt1}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Use safe JSON parsing
        analysis1 = safe_json_parse(response1.choices[0].message.content)
        if analysis1 is None:
            st.error("Failed to parse first analysis response")
            return None
        
        # Second LLM call: Determine DISC type based on personality traits
        disc_df = load_disc_types()
        disc_types_text = disc_df.to_string(index=False)
        
        prompt2 = f"""
        Based on the personality analysis below, determine the DISC type for this person.
        
        Available DISC types:
        {disc_types_text}
        
        Personality Analysis:
        {json.dumps(analysis1, indent=2)}
        
        Return ONLY a valid JSON object with the following structure:
        {{
            "disc_profile": {{
                "primary_type": "D",
                "secondary_type": "I", 
                "full_type": "DI",
                "crystal_label": "Driver",
                "description": "Description of this DISC type",
                "confidence_score": 85
            }},
            "sales_approach_recommendations": [
                "recommendation1",
                "recommendation2", 
                "recommendation3"
            ]
        }}
        
        Return only the JSON object, no additional text or explanation.
        """
        
        response2 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt2}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Use safe JSON parsing
        analysis2 = safe_json_parse(response2.choices[0].message.content)
        if analysis2 is None:
            st.error("Failed to parse second analysis response")
            return None
        
        # Combine the analyses
        combined_analysis = {
            "name": analysis1.get("name", "Unknown"),
            "position": analysis1.get("position", "Not Available"),
            "company": analysis1.get("company", "Not Available"),
            "location": analysis1.get("location", "Not Available"),
            "personality_traits": analysis1.get("personality_traits", {}),
            "disc_profile": analysis2.get("disc_profile", {}),
            "key_insights": analysis1.get("key_insights", []),
            "communication_preferences": analysis1.get("communication_preferences", []),
            "sales_approach_recommendations": analysis2.get("sales_approach_recommendations", [])
        }
        
        return combined_analysis
        
    except Exception as e:
        st.error(f"Error analyzing profile: {e}")
        return None

def generate_personality_traits_chart(traits: Dict[str, float]):
    """Generate personality traits visualization chart"""
    
    trait_names = [
        "Risk Tolerant ↔ Risk Averse",
        "Trusting ↔ Skeptical", 
        "Optimistic ↔ Pragmatic",
        "Deliberate ↔ Fast-paced",
        "Matter-of-fact ↔ Expressive",
        "Autonomous ↔ Collaborative",
        "Supporting ↔ Dominant"
    ]
    
    trait_values = [
        traits.get("risk_tolerance", 50),
        traits.get("trust_level", 50),
        traits.get("optimism_level", 50),
        traits.get("decision_style", 50),
        traits.get("communication_style", 50),
        traits.get("work_style", 50),
        traits.get("leadership_style", 50)
    ]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    for i, (name, value) in enumerate(zip(trait_names, trait_values)):
        # Create a horizontal bar
        fig.add_trace(go.Bar(
            y=[name],
            x=[100],  # Full width
            orientation='h',
            marker=dict(
                color='rgba(212, 137, 106, 0.1)',
                line=dict(color='rgba(212, 137, 106, 0.3)', width=1)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add the PP marker
        fig.add_trace(go.Scatter(
            x=[value],
            y=[name],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=15,
                color='#d4896a',
                line=dict(color='white', width=2)
            ),
            text=['PP'],
            textposition='middle center',
            textfont=dict(color='white', size=10, family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="Personality Traits",
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickmode='linear',
            tick0=0,
            dtick=20,
            ticktext=['0', '20', '40', '60', '80', '100'],
            tickvals=[0, 20, 40, 60, 80, 100]
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=10)
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

def generate_sales_pitch(analysis: Dict[str, Any], solution_description: str) -> str:
    """Generate personalized sales pitch based on personality and DISC analysis"""
    
    # Extract detailed personality traits for better pitch customization
    traits = analysis.get('personality_traits', {})
    disc_profile = analysis.get('disc_profile', {})
    
    # Create detailed personality summary for the pitch
    personality_summary = f"""
    DISC Profile: {disc_profile.get('crystal_label', 'Unknown')} ({disc_profile.get('full_type', 'Unknown')})
    - {disc_profile.get('description', 'Not Available')}
    
    Personality Traits (0-100 scale):
    - Risk Tolerance: {traits.get('risk_tolerance', 50)} (0=Risk Tolerant, 100=Risk Averse)
    - Trust Level: {traits.get('trust_level', 50)} (0=Trusting, 100=Skeptical)
    - Optimism: {traits.get('optimism_level', 50)} (0=Optimistic, 100=Pragmatic)
    - Decision Style: {traits.get('decision_style', 50)} (0=Deliberate, 100=Fast-paced)
    - Communication: {traits.get('communication_style', 50)} (0=Matter-of-fact, 100=Expressive)
    - Work Style: {traits.get('work_style', 50)} (0=Autonomous, 100=Collaborative)
    - Leadership: {traits.get('leadership_style', 50)} (0=Supporting, 100=Dominant)
    """
    
    prompt = f"""
    You are an expert sales professional at ManPower Group with deep expertise in personality-based selling. Create a highly personalized 2-minute elevator sales pitch that strategically leverages the client's DISC profile and personality traits.

    CLIENT PROFILE:
    - Name: {analysis.get('name', 'Unknown')}
    - Position: {analysis.get('position', 'Not Available')}
    - Company: {analysis.get('company', 'Not Available')}
    - Location: {analysis.get('location', 'Not Available')}

    DETAILED PERSONALITY ANALYSIS:
    {personality_summary}

    KEY BEHAVIORAL INSIGHTS:
    {chr(10).join(f"• {insight}" for insight in analysis.get('key_insights', []))}

    COMMUNICATION PREFERENCES:
    {chr(10).join(f"• {pref}" for pref in analysis.get('communication_preferences', []))}

    RECOMMENDED SALES APPROACH:
    {chr(10).join(f"• {rec}" for rec in analysis.get('sales_approach_recommendations', []))}

    CLIENT'S REQUIREMENTS & CONTEXT:
    {solution_description}

    INSTRUCTIONS:
    Create a professional, compelling 2-minute elevator pitch (approximately 250-300 words) that:

    1. DISC-ADAPTED COMMUNICATION:
       - Match their communication style based on DISC type
       - Use language patterns that resonate with their personality
       - Adjust pace and detail level to their preferences

    2. PERSONALITY-DRIVEN VALUE PROPOSITION:
       - Frame benefits according to their risk tolerance
       - Appeal to their decision-making style (deliberate vs fast-paced)
       - Address their trust level appropriately
       - Match their optimism/pragmatism balance

    3. BEHAVIORAL CUSTOMIZATION:
       - Adapt to their work style (autonomous vs collaborative)
       - Consider their leadership preferences
       - Use examples that align with their industry and role

    4. PROFESSIONAL STRUCTURE:
       - Strong opening that captures attention based on their communication style
       - Clear value proposition aligned with their personality
       - Specific benefits tailored to their traits
       - Compelling call-to-action that matches their decision style

    5. STRATEGIC ELEMENTS:
       - Use data/facts for high skepticism scores
       - Include emotional elements for expressive communicators
       - Emphasize stability for risk-averse profiles
       - Highlight innovation for risk-tolerant personalities

    Generate a pitch that feels natural and conversational while being strategically crafted for maximum impact based on their unique personality profile. Do not explicitly mention DISC types or personality scores in the pitch itself.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating sales pitch: {e}")
        return "Error generating sales pitch."

def save_session_data(analysis: Dict[str, Any], solution: str, pitch: str):
    """Save session data to JSON file - silent save without UI notification"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sessions/profile_{analysis.get('name', 'unknown').replace(' ', '_')}_{timestamp}.json"
    
    data = {
        "name": analysis.get('name', 'Unknown'),
        "position": analysis.get('position', 'Not Available'),
        "company": analysis.get('company', 'Not Available'),
        "location": analysis.get('location', 'Not Available'),
        "personality_traits": analysis.get('personality_traits', {}),
        "disc_profile": analysis.get('disc_profile', {}),
        "key_insights": analysis.get('key_insights', []),
        "communication_preferences": analysis.get('communication_preferences', []),
        "sales_approach_recommendations": analysis.get('sales_approach_recommendations', []),
        "solution_description": solution,
        "sales_pitch": pitch,
        "timestamp": timestamp
    }
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        # Silent save - no UI notification
    except Exception as e:
        st.error(f"Error saving session: {e}")

# Streamlit UI
def main():
    
    # Page header with consistent styling
    st.markdown('<div class="arsenal-title">Personality-Driven Sales Pitch Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="arsenal-subtitle">Upload a LinkedIn profile and generate personalized sales pitches based on personality analysis and DISC profiling</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if "profile_analysis" not in st.session_state:
        st.session_state.profile_analysis = None
    if "profile_uploaded" not in st.session_state:
        st.session_state.profile_uploaded = False
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "solution_description" not in st.session_state:
        st.session_state.solution_description = ""
    if "sales_pitch" not in st.session_state:
        st.session_state.sales_pitch = ""
    
    st.divider()
    
    # Section 1: Upload Profile
    st.markdown('<h3 style="color: #374151; font-family: var(--header-font);">Upload LinkedIn Profile</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload LinkedIn Profile (PDF)",
        type="pdf",
        help="Upload a PDF export of the LinkedIn profile you want to analyze"
    )
    
    if uploaded_file and not st.session_state.profile_uploaded:
        with st.spinner("Extracting profile information using LlamaParse..."):
            profile_text = extract_text_from_pdf(uploaded_file)
            if profile_text:
                name, position, company = extract_basic_info(profile_text)
                
                # Store in session state first
                st.session_state.profile_text = profile_text
                st.session_state.extracted_name = name
                st.session_state.extracted_position = position
                st.session_state.extracted_company = company
                st.session_state.profile_uploaded = True
                
        # Display extracted info after storing in session state
        if st.session_state.profile_uploaded:
            st.markdown("### Profile Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="pillar-description">
                    <strong>Name:</strong><br>{st.session_state.extracted_name}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="pillar-description">
                    <strong>Position:</strong><br>{st.session_state.extracted_position}
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="pillar-description">
                    <strong>Company:</strong><br>{st.session_state.extracted_company}
                </div>
                """, unsafe_allow_html=True)
            
            # Start New Analysis button after file upload
            st.divider()
            if st.button("Start New Analysis", use_container_width=True, type="secondary"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Section 2: Analyze Personality (Auto-triggered after upload)
    if st.session_state.profile_uploaded and not st.session_state.analysis_complete:
        st.divider()
        st.markdown('<div style="text-align: center;"><h3 style="color: #374151; font-family: var(--header-font);">Analyzing Personality</h3></div>', unsafe_allow_html=True)
        
        # Show progress bar during analysis
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Starting personality analysis...")
        progress_bar.progress(20)
        
        status_text.text("Analyzing communication style and traits...")
        progress_bar.progress(50)
        
        analysis = analyze_linkedin_profile(st.session_state.profile_text)
        
        if analysis:
            status_text.text("Determining DISC profile...")
            progress_bar.progress(80)
            
            status_text.text("Analysis complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Store results in session state
            st.session_state.profile_analysis = analysis
            st.session_state.analysis_complete = True
    
    # Section 3: Display Analysis Results
    if st.session_state.analysis_complete and st.session_state.profile_analysis:
        st.divider()
        st.markdown('<div style="text-align: center;"><h3 style="color: #374151; font-family: var(--header-font);">Personality Analysis Results</h3></div>', unsafe_allow_html=True)
        
        analysis = st.session_state.profile_analysis
        
        # Display results with clean layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h4 style="color: #374151;">Personality Traits</h4>', unsafe_allow_html=True)
            fig = generate_personality_traits_chart(analysis.get('personality_traits', {}))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h4 style="color: #374151;">DISC Profile</h4>', unsafe_allow_html=True)
            disc_profile = analysis.get('disc_profile', {})
            st.markdown(f"""
            <div class="pillar-description">
                <strong>Type:</strong> {disc_profile.get('crystal_label', 'Unknown')}<br>
                <strong>Code:</strong> {disc_profile.get('full_type', 'Unknown')}<br>
                <strong>Confidence:</strong> {disc_profile.get('confidence_score', 0)}%<br><br>
                <strong>Description:</strong><br>
                {disc_profile.get('description', 'Not Available')}
            </div>
            """, unsafe_allow_html=True)
        
        # Key insights in clean format
        st.markdown('<h4 style="color: #374151;">Key Insights</h4>', unsafe_allow_html=True)
        insights = analysis.get('key_insights', [])
        if insights:
            insights_text = "\n".join(f"• {insight}" for insight in insights)
            st.markdown(f"""
            <div class="pillar-description">
                {insights_text.replace(chr(10), "<br>")}
            </div>
            """, unsafe_allow_html=True)
        
        # Communication preferences
        st.markdown('<h4 style="color: #374151;">Communication Preferences</h4>', unsafe_allow_html=True)
        prefs = analysis.get('communication_preferences', [])
        if prefs:
            prefs_text = "\n".join(f"• {pref}" for pref in prefs)
            st.markdown(f"""
            <div class="pillar-description">
                {prefs_text.replace(chr(10), "<br>")}
            </div>
            """, unsafe_allow_html=True)
    
    # Section 4: Generate Sales Pitch
    if st.session_state.analysis_complete and st.session_state.profile_analysis:
        st.divider()
        st.markdown('<div style="text-align: center;"><h3 style="color: #374151; font-family: var(--header-font);">Generate Sales Pitch</h3></div>', unsafe_allow_html=True)
        
        # Solution description input
        solution_description = st.text_area(
            "Describe the client's requirements and your solution:",
            value=st.session_state.solution_description,
            height=150,
            placeholder="Describe the client's organization, specific requirements, and the HR solution you're offering..."
        )
        
        if st.button("Generate Personalized Pitch", type="primary", use_container_width=True):
            with st.spinner("Generating personalized sales pitch..."):
                pitch = generate_sales_pitch(st.session_state.profile_analysis, solution_description)
                st.session_state.sales_pitch = pitch
                st.session_state.solution_description = solution_description
            
            # Display the pitch with clean styling
            st.markdown('<h4 style="color: #374151;">Personalized Sales Pitch</h4>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="pillar-description">
                {pitch}
            </div>
            """, unsafe_allow_html=True)
            
            # Save session silently (no UI notification)
            save_session_data(st.session_state.profile_analysis, solution_description, pitch)
        
        # Show existing pitch if available
        elif st.session_state.sales_pitch:
            st.markdown('<h4 style="color: #374151;">Personalized Sales Pitch</h4>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="pillar-description">
                {st.session_state.sales_pitch}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
