import streamlit as st
import tempfile
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from llama_cloud_services import LlamaParse
import pandas as pd
from openai import OpenAI
import plotly.graph_objects as go

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
        st.session_state.job_description_text = None
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = None
    if 'parsed_job_info' not in st.session_state:
        st.session_state.parsed_job_info = None
    if 'parsed_resume_info' not in st.session_state:
        st.session_state.parsed_resume_info = None
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = None
    if 'compatibility_assessment' not in st.session_state:
        st.session_state.compatibility_assessment = None
    if 'parsing_complete' not in st.session_state:
        st.session_state.parsing_complete = False

def validate_api_keys() -> tuple[bool, bool]:
    """Validate API keys"""
    llamacloud_valid = LLAMACLOUD_API_KEY and LLAMACLOUD_API_KEY != "your-llamacloud-api-key" and LLAMACLOUD_API_KEY.startswith("llx-")
    openrouter_valid = OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your-openrouter-api-key"
    return llamacloud_valid, openrouter_valid

def render_api_key_error():
    """Render API key configuration error message"""
    llamacloud_valid, openrouter_valid = validate_api_keys()
    
    if not llamacloud_valid:
        st.error("**Configuration Required:** LlamaCloud API Key is missing or invalid.")
        
        with st.expander("LlamaCloud Setup Instructions", expanded=True):
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
            """)
    
    if not openrouter_valid:
        st.error("**Configuration Required:** OpenRouter API Key is missing or invalid.")
        
        with st.expander("OpenRouter Setup Instructions", expanded=True):
            st.markdown("""
            ### How to get your OpenRouter API Key:
            
            1. **Sign up** for an account at [OpenRouter](https://openrouter.ai)
            2. **Navigate** to the API Keys section
            3. **Generate** a new API key
            4. **Copy** the key
            
            ### Add the key to your Streamlit secrets:
            
            Create or edit `.streamlit/secrets.toml` in your project root:
            
            ```toml
            OPENROUTER_API_KEY = "your-actual-openrouter-api-key-here"
            ```
            """)

def parse_document_with_llamacloud(uploaded_file) -> Optional[str]:
    """Parse document using LlamaCloud Parse API"""
    try:
        parser_config = {
            "api_key": LLAMACLOUD_API_KEY,
            "result_type": "markdown",
            "verbose": True,
            "language": "en"
        }
        
        parser = LlamaParse(**parser_config)
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Parse the document
        result = parser.parse(tmp_file_path)
        
        # Extract markdown content
        markdown_docs = result.get_markdown_documents(split_by_page=False)
        content = "\n\n".join([doc.text for doc in markdown_docs]) if markdown_docs else ""
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return content
        
    except Exception as e:
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        st.error(f"Error parsing document: {str(e)}")
        return None

def extract_job_information(job_description: str) -> Optional[Dict[str, Any]]:
    """Extract structured information from job description using OpenRouter API"""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        prompt = f"""
        Analyze the following job description and extract key information in a structured format.
        
        Job Description:
        {job_description}
        
        Please provide a JSON response with the following structure:
        {{
            "job_title": "extracted job title",
            "company": "company name if mentioned",
            "role_level": "entry/mid/senior/executive/leadership",
            "role_type": "technical/managerial/leadership/hybrid",
            "key_responsibilities": ["list of main responsibilities"],
            "required_skills": ["list of required technical skills"],
            "soft_skills": ["list of required soft skills"],
            "experience_required": "years of experience needed",
            "industry": "industry sector",
            "key_qualifications": ["list of educational/certification requirements"],
            "company_culture": "description of company culture if mentioned",
            "growth_opportunities": "career growth aspects if mentioned"
        }}
        
        Extract only information that is explicitly mentioned or can be reasonably inferred from the job description.
        """

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Interview Coach",
            },
            model="google/gemini-2.5-flash-lite",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR analyst. Extract structured information from job descriptions accurately and comprehensively."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )
        
        response_content = completion.choices[0].message.content
        
        # Extract JSON from response
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_content = response_content[start_idx:end_idx]
            return json.loads(json_content)
        
        return None
        
    except Exception as e:
        st.error(f"Error extracting job information: {str(e)}")
        return None

def extract_resume_information(resume_text: str) -> Optional[Dict[str, Any]]:
    """Extract structured information from resume using OpenRouter API"""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        prompt = f"""
        Analyze the following resume and extract key information in a structured format.
        
        Resume:
        {resume_text}
        
        Please provide a JSON response with the following structure:
        {{
            "candidate_name": "name if mentioned",
            "current_title": "current job title",
            "total_experience": "total years of experience",
            "technical_skills": ["list of technical skills"],
            "soft_skills": ["list of demonstrated soft skills"],
            "work_experience": [
                {{
                    "title": "job title",
                    "company": "company name",
                    "duration": "time period",
                    "key_achievements": ["list of key achievements"]
                }}
            ],
            "education": ["list of educational qualifications"],
            "certifications": ["list of certifications"],
            "projects": ["list of notable projects"],
            "leadership_experience": ["list of leadership roles/experiences"],
            "industry_experience": ["list of industries worked in"],
            "career_progression": "description of career growth pattern"
        }}
        
        Extract only information that is explicitly mentioned in the resume.
        """

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Interview Coach",
            },
            model="google/gemini-2.5-flash-lite",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert resume analyst. Extract structured information from resumes accurately and comprehensively."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )
        
        response_content = completion.choices[0].message.content
        
        # Extract JSON from response
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_content = response_content[start_idx:end_idx]
            return json.loads(json_content)
        
        return None
        
    except Exception as e:
        st.error(f"Error extracting resume information: {str(e)}")
        return None

def generate_interview_questions(job_info: Dict[str, Any], resume_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate comprehensive interview questions using OpenRouter API"""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        # Determine role type for customized questions
        role_level = job_info.get('role_level', 'mid')
        role_type = job_info.get('role_type', 'technical')
        
        prompt = f"""
        Based on the job requirements and candidate's background, generate comprehensive interview questions in three categories.
        
        Job Information:
        {json.dumps(job_info, indent=2)}
        
        Candidate Background:
        {json.dumps(resume_info, indent=2)}
        
        Generate questions in the following format:
        {{
            "behavioral_questions": [
                {{
                    "question": "Resume-based behavioral question",
                    "sample_answer": "Example of a strong answer",
                    "evaluation_criteria": "What interviewers look for"
                }}
            ],
            "technical_experience": [
                {{
                    "question": "Technical question based on candidate's experience relevant to the job",
                    "sample_answer": "Example technical response",
                    "evaluation_criteria": "Technical competencies being assessed"
                }}
            ],
            "job_specific_technical": {{
                "high_difficulty": [
                    {{
                        "question": "Advanced technical question for the role",
                        "sample_answer": "Comprehensive technical answer",
                        "evaluation_criteria": "Advanced skills assessment"
                    }}
                ],
                "medium_difficulty": [
                    {{
                        "question": "Intermediate technical question",
                        "sample_answer": "Good technical response",
                        "evaluation_criteria": "Core competency assessment"
                    }}
                ],
                "low_difficulty": [
                    {{
                        "question": "Basic technical question",
                        "sample_answer": "Fundamental technical answer",
                        "evaluation_criteria": "Basic understanding assessment"
                    }}
                ]
            }}
        }}

        Guidelines for question generation:
        1. **Behavioral Questions (5 questions)**: Focus on specific experiences from the candidate's resume that relate to the job requirements. Use STAR method framework.
        
        2. **Technical Experience Questions (5 questions)**: Based on the candidate's actual technical experience but relevant to the new role. Bridge their past experience with job requirements.
        
        3. **Job-Specific Technical Questions**: 
           - **High (3 questions)**: Advanced concepts, system design, architecture decisions
           - **Medium (4 questions)**: Core technical skills, problem-solving
           - **Low (3 questions)**: Fundamental concepts, basic implementation
        
        **Role-Specific Adaptations**:
        - For **Leadership/Executive roles**: Include strategic thinking, team management, organizational impact questions
        - For **Managerial roles**: Add people management, project coordination, stakeholder communication
        - For **Technical roles**: Focus on technical depth, problem-solving methodology, innovation
        - For **Hybrid roles**: Balance technical and people management aspects
        
        **Experience Level Adaptations**:
        - **Entry level**: Focus on learning ability, foundational knowledge, growth potential
        - **Mid level**: Balance experience application with growth areas
        - **Senior level**: Leadership impact, mentoring, strategic contributions
        - **Executive level**: Vision, organizational transformation, industry leadership
        
        Ensure all questions are:
        - Directly relevant to both the job requirements and candidate background
        - Progressive in complexity within each category
        - Include specific examples and scenarios
        - Provide actionable sample answers
        - Include clear evaluation criteria for interviewers
        """

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Interview Coach",
            },
            model="google/gemini-2.5-flash-lite",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert interview coach and HR professional with deep expertise in creating role-specific, candidate-tailored interview questions across all seniority levels and role types."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        
        response_content = completion.choices[0].message.content
        
        # Extract JSON from response
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_content = response_content[start_idx:end_idx]
            return json.loads(json_content)
        
        return None
        
    except Exception as e:
        st.error(f"Error generating interview questions: {str(e)}")
        return None

def assess_job_compatibility(job_info: Dict[str, Any], resume_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Assess candidate compatibility on 5 pillars using OpenRouter API"""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        prompt = f"""
        Analyze the candidate's compatibility with the job requirements and provide ratings on a scale of 1-5 for each pillar.
        
        Job Requirements:
        {json.dumps(job_info, indent=2)}
        
        Candidate Profile:
        {json.dumps(resume_info, indent=2)}
        
        Please provide a JSON response with the following structure:
        {{
            "compatibility_scores": {{
                "skills": {{
                    "score": 4,
                    "explanation": "Detailed explanation of skills match"
                }},
                "experience": {{
                    "score": 3,
                    "explanation": "Detailed explanation of experience match"
                }},
                "education": {{
                    "score": 5,
                    "explanation": "Detailed explanation of education match"
                }},
                "industry_match": {{
                    "score": 4,
                    "explanation": "Detailed explanation of industry alignment"
                }},
                "culture_match": {{
                    "score": 3,
                    "explanation": "Detailed explanation of culture fit based on available information"
                }}
            }},
            "overall_fit_summary": "Comprehensive 2-3 paragraph summary of candidate fit, including strengths, areas for development, and specific recommendations"
        }}
        
        Rating Guidelines:
        - **Skills (1-5)**: How well do the candidate's technical and soft skills match the job requirements?
        - **Experience (1-5)**: How well does the years and type of experience align with job needs?
        - **Education (1-5)**: How well does educational background match requirements?
        - **Industry Match (1-5)**: How relevant is the candidate's industry experience to the target role?
        - **Culture Match (1-5)**: Based on available info, how well might the candidate fit the company culture?
        
        Scoring Scale:
        - 1: Poor match, significant gaps
        - 2: Below average match, notable gaps
        - 3: Average match, some gaps
        - 4: Good match, minor gaps
        - 5: Excellent match, very strong alignment
        
        Provide specific, actionable insights in explanations and overall summary.
        """

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Interview Coach",
            },
            model="google/gemini-2.5-flash-lite",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR analyst and career counselor. Provide accurate, constructive compatibility assessments that help candidates understand their fit and areas for improvement."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )
        
        response_content = completion.choices[0].message.content
        
        # Extract JSON from response
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_content = response_content[start_idx:end_idx]
            return json.loads(json_content)
        
        return None
        
    except Exception as e:
        st.error(f"Error assessing job compatibility: {str(e)}")
        return None

def create_compatibility_radar_chart(compatibility_data: Dict[str, Any]) -> go.Figure:
    """Create a radar chart for compatibility scores"""
    scores = compatibility_data.get('compatibility_scores', {})
    
    # Extract scores and labels
    categories = ['Skills', 'Experience', 'Education', 'Industry Match', 'Culture Match']
    values = [
        scores.get('skills', {}).get('score', 0),
        scores.get('experience', {}).get('score', 0),
        scores.get('education', {}).get('score', 0),
        scores.get('industry_match', {}).get('score', 0),
        scores.get('culture_match', {}).get('score', 0)
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(212, 137, 106, 0.3)',
        line=dict(color='rgb(212, 137, 106)', width=3),
        marker=dict(size=8, color='rgb(212, 137, 106)'),
        name='Compatibility Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['1', '2', '3', '4', '5'],
                gridcolor='rgba(0, 0, 0, 0.1)',
                linecolor='rgba(0, 0, 0, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(0, 0, 0, 0.1)',
                linecolor='rgba(0, 0, 0, 0.2)'
            )
        ),
        showlegend=False,
        title=dict(
            text="Job Compatibility Assessment",
            x=0.5,
            font=dict(size=18, color='#374151')
        ),
        font=dict(size=12, color='#374151'),
        #plot_bgcolor='rgba(240, 238, 230, 1)',
        #paper_bgcolor='rgba(240, 238, 230, 1)',
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return fig

def render_compatibility_assessment():
    """Render the job compatibility assessment section"""
    if not (st.session_state.get('compatibility_assessment') and 
            st.session_state.parsed_job_info and 
            st.session_state.parsed_resume_info):
        return
    
    st.markdown('<div class="arsenal-title">Job Compatibility Assessment</div>', unsafe_allow_html=True)
    
    compatibility_data = st.session_state.compatibility_assessment
    scores = compatibility_data.get('compatibility_scores', {})
    
    # Radar Chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_compatibility_radar_chart(compatibility_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Compatibility Scores")
        
        # Display scores with explanations
        for pillar, data in scores.items():
            pillar_name = pillar.replace('_', ' ').title()
            score = data.get('score', 0)
            
            # Color coding for scores
            if score >= 4:
                color = "#10b981"  # Green
            elif score >= 3:
                color = "#f59e0b"  # Yellow
            else:
                color = "#ef4444"  # Red
            
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="color: {color};">{pillar_name}: {score}/5</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Overall score
        avg_score = sum(data.get('score', 0) for data in scores.values()) / len(scores) if scores else 0
        if avg_score >= 4:
            overall_color = "#10b981"
            overall_status = "Strong Match"
        elif avg_score >= 3:
            overall_color = "#f59e0b"
            overall_status = "Good Match"
        else:
            overall_color = "#ef4444"
            overall_status = "Needs Development"
        
        st.markdown(f"""
        <div class="pillar-description" style="text-align: center; margin-top: 1rem;">
            <strong style="color: {overall_color};">Overall Compatibility</strong><br>
            <span style="font-size: 1.5rem; color: {overall_color};">{avg_score:.1f}/5</span><br>
            <em style="color: {overall_color};">{overall_status}</em>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed explanations
    st.markdown("#### Detailed Assessment")
    
    for pillar, data in scores.items():
        pillar_name = pillar.replace('_', ' ').title()
        score = data.get('score', 0)
        explanation = data.get('explanation', 'No explanation available')
        
        with st.expander(f"{pillar_name} - {score}/5"):
            st.markdown(explanation)
    
    # Am I a good fit? section
    st.markdown("#### Am I a Good Fit?")
    
    overall_summary = compatibility_data.get('overall_fit_summary', 'Assessment not available')
    
    st.markdown(f"""
    <div class="pillar-description">
        {overall_summary}
    </div>
    """, unsafe_allow_html=True)

def render_header():
    """Render page header with consistent styling"""
    st.markdown('<div class="hero-title">Job Compatibility & Interview</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Preparation Coach</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">AI-powered interview preparation tailored to your background and target role</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-description">
    Paste the job description and upload your resume to generate personalized interview questions. 
    Get behavioral questions based on your experience, technical questions relevant to your background, 
    and role-specific questions at different difficulty levels.
    </div>
    """, unsafe_allow_html=True)

def render_file_upload():
    """Render file upload interface"""
    st.markdown('<div class="arsenal-title">Input Documents</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Job Description")
        job_text = st.text_area(
            "Paste the job description here",
            height=200,
            key="job_text",
            help="Paste the complete job description or job posting here",
            placeholder="Copy and paste the job description here..."
        )
    
    with col2:
        st.markdown("#### Resume")
        resume_file = st.file_uploader(
            "Upload your resume (PDF)",
            type=['pdf'],
            key="resume_file",
            help="Upload your current resume as a PDF file"
        )
    
    return job_text, resume_file

def render_extracted_info():
    """Render extracted information from documents"""
    if st.session_state.parsed_job_info or st.session_state.parsed_resume_info:
        st.markdown('<div class="arsenal-title">Extracted Information</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.parsed_job_info:
                st.markdown("#### Job Analysis")
                job_info = st.session_state.parsed_job_info
                
                st.markdown(f"""
                <div class="pillar-description">
                    <strong>Position:</strong> {job_info.get('job_title', 'Not specified')}<br>
                    <strong>Company:</strong> {job_info.get('company', 'Not specified')}<br>
                    <strong>Level:</strong> {job_info.get('role_level', 'Not specified').title()}<br>
                    <strong>Type:</strong> {job_info.get('role_type', 'Not specified').title()}<br>
                    <strong>Experience:</strong> {job_info.get('experience_required', 'Not specified')}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Detailed Job Analysis"):
                    st.json(job_info)
        
        with col2:
            if st.session_state.parsed_resume_info:
                st.markdown("#### Resume Analysis")
                resume_info = st.session_state.parsed_resume_info
                
                st.markdown(f"""
                <div class="pillar-description">
                    <strong>Candidate:</strong> {resume_info.get('candidate_name', 'Not specified')}<br>
                    <strong>Current Role:</strong> {resume_info.get('current_title', 'Not specified')}<br>
                    <strong>Experience:</strong> {resume_info.get('total_experience', 'Not specified')}<br>
                    <strong>Industries:</strong> {', '.join(resume_info.get('industry_experience', [])) if resume_info.get('industry_experience') else 'Not specified'}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Detailed Resume Analysis"):
                    st.json(resume_info)

def render_questions_display():
    """Render generated interview questions"""
    if not st.session_state.generated_questions:
        return
    
    st.markdown('<div class="arsenal-title">Interview Questions</div>', unsafe_allow_html=True)
    
    questions = st.session_state.generated_questions
    
    # Behavioral Questions
    if 'behavioral_questions' in questions:
        st.markdown("### Resume-Based Behavioral Questions")
        st.markdown("Questions based on your specific experience and background")
        
        for i, q in enumerate(questions['behavioral_questions'], 1):
            with st.expander(f"Behavioral Question {i}"):
                st.markdown(f"**Question:** {q.get('question', '')}")
                with st.expander("Sample Answer"):
                    st.markdown(f"{q.get('sample_answer', '')}")
                    st.markdown(f"**Evaluation Criteria:** {q.get('evaluation_criteria', '')}")
        
        st.divider()
    
    # Technical Experience Questions
    if 'technical_experience' in questions:
        st.markdown("### Technical Questions (Based on Your Experience)")
        st.markdown("Technical questions relevant to your background and the target role")
        
        for i, q in enumerate(questions['technical_experience'], 1):
            with st.expander(f"Technical Experience Question {i}"):
                st.markdown(f"**Question:** {q.get('question', '')}")
                with st.expander("Sample Answer"):
                    st.markdown(f"{q.get('sample_answer', '')}")
                    st.markdown(f"**Evaluation Criteria:** {q.get('evaluation_criteria', '')}")
        
        st.divider()
    
    # Job-Specific Technical Questions
    if 'job_specific_technical' in questions:
        st.markdown("### Job-Specific Technical Questions")
        
        job_tech = questions['job_specific_technical']
        
        # Low Difficulty
        if 'low_difficulty' in job_tech:
            st.markdown("#### Low Difficulty")
            st.markdown("Fundamental concepts and basic implementation")
            for i, q in enumerate(job_tech['low_difficulty'], 1):
                with st.expander(f"Low Difficulty Question {i}"):
                    st.markdown(f"**Question:** {q.get('question', '')}")
                    with st.expander("Sample Answer"):
                        st.markdown(f"{q.get('sample_answer', '')}")
                        st.markdown(f"**Evaluation Criteria:** {q.get('evaluation_criteria', '')}")
        
        # Medium Difficulty
        if 'medium_difficulty' in job_tech:
            st.markdown("#### Medium Difficulty")
            st.markdown("Core technical skills and problem-solving")
            for i, q in enumerate(job_tech['medium_difficulty'], 1):
                with st.expander(f"Medium Difficulty Question {i}"):
                    st.markdown(f"**Question:** {q.get('question', '')}")
                    with st.expander("Sample Answer"):
                        st.markdown(f"{q.get('sample_answer', '')}")
                        st.markdown(f"**Evaluation Criteria:** {q.get('evaluation_criteria', '')}")

        # High Difficulty
        if 'high_difficulty' in job_tech:
            st.markdown("#### High Difficulty")
            st.markdown("Advanced technical concepts and system design")
            for i, q in enumerate(job_tech['high_difficulty'], 1):
                with st.expander(f"High Difficulty Question {i}"):
                    st.markdown(f"**Question:** {q.get('question', '')}")
                    with st.expander("Sample Answer"):
                        st.markdown(f"{q.get('sample_answer', '')}")
                        st.markdown(f"**Evaluation Criteria:** {q.get('evaluation_criteria', '')}")   


def render_download_section():
    """Render download section for questions"""
    if not st.session_state.generated_questions:
        return
    
    st.divider()
    st.markdown("### Download Questions")
    
    # Create downloadable content
    questions = st.session_state.generated_questions
    job_info = st.session_state.parsed_job_info or {}
    resume_info = st.session_state.parsed_resume_info or {}
    
    download_content = f"""Interview Preparation Questions
{'='*50}

Position: {job_info.get('job_title', 'Not specified')}
Company: {job_info.get('company', 'Not specified')}
Candidate: {resume_info.get('candidate_name', 'Not specified')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*50}

BEHAVIORAL QUESTIONS (Resume-Based)
{'-'*35}
"""
    
    if 'behavioral_questions' in questions:
        for i, q in enumerate(questions['behavioral_questions'], 1):
            download_content += f"""
Q{i}: {q.get('question', '')}

Sample Answer: {q.get('sample_answer', '')}

Evaluation Criteria: {q.get('evaluation_criteria', '')}

{'-'*30}
"""
    
    download_content += f"""

TECHNICAL QUESTIONS (Experience-Based)
{'-'*37}
"""
    
    if 'technical_experience' in questions:
        for i, q in enumerate(questions['technical_experience'], 1):
            download_content += f"""
Q{i}: {q.get('question', '')}

Sample Answer: {q.get('sample_answer', '')}

Evaluation Criteria: {q.get('evaluation_criteria', '')}

{'-'*30}
"""
    
    download_content += f"""

JOB-SPECIFIC TECHNICAL QUESTIONS
{'-'*33}
"""
    
    if 'job_specific_technical' in questions:
        job_tech = questions['job_specific_technical']
        
        for difficulty in ['high_difficulty', 'medium_difficulty', 'low_difficulty']:
            if difficulty in job_tech:
                download_content += f"""

{difficulty.replace('_', ' ').title()}:
"""
                for i, q in enumerate(job_tech[difficulty], 1):
                    download_content += f"""
Q{i}: {q.get('question', '')}

Sample Answer: {q.get('sample_answer', '')}

Evaluation Criteria: {q.get('evaluation_criteria', '')}

{'-'*30}
"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="Download Interview Questions",
            data=download_content,
            file_name=f"interview_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Check API configuration
    llamacloud_valid, openrouter_valid = validate_api_keys()
    if not llamacloud_valid or not openrouter_valid:
        render_api_key_error()
        st.stop()
    
    st.divider()
    
    # File upload section
    job_text, resume_file = render_file_upload()
    
    # Process uploaded files
    if job_text and job_text.strip() and resume_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Use job description text directly
                    status_text.text("Processing job description...")
                    progress_bar.progress(20)
                    st.session_state.job_description_text = job_text
                    
                    # Parse resume
                    status_text.text("Extracting resume...")
                    progress_bar.progress(40)
                    resume_text = parse_document_with_llamacloud(resume_file)
                    
                    if resume_text:
                        st.session_state.resume_text = resume_text
                        
                        # Extract job information
                        status_text.text("Analyzing job requirements...")
                        progress_bar.progress(60)
                        job_info = extract_job_information(job_text)
                        
                        if job_info:
                            st.session_state.parsed_job_info = job_info
                            
                            # Extract resume information
                            status_text.text("Analyzing resume...")
                            progress_bar.progress(80)
                            resume_info = extract_resume_information(resume_text)
                            
                            if resume_info:
                                st.session_state.parsed_resume_info = resume_info
                                st.session_state.parsing_complete = True
                                
                                progress_bar.progress(100)
                                status_text.text("Processing complete!")
                                
                                # Clear progress indicators
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.success("Documents processed successfully!")
    
    # Show extracted information
    if st.session_state.parsing_complete:
        st.divider()
        render_extracted_info()
        
        # Compatibility Assessment button
        if st.session_state.parsed_job_info and st.session_state.parsed_resume_info:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Assess Job Compatibility", type="primary", use_container_width=True):
                    with st.spinner("Analyzing job compatibility..."):
                        compatibility = assess_job_compatibility(
                            st.session_state.parsed_job_info,
                            st.session_state.parsed_resume_info
                        )
                        
                        if compatibility:
                            st.session_state.compatibility_assessment = compatibility
                            st.success("Compatibility assessment completed!")
                        else:
                            st.error("Failed to assess compatibility. Please try again.")
    
    # Show compatibility assessment
    if st.session_state.compatibility_assessment:
        st.divider()
        render_compatibility_assessment()
        
        # Generate questions button
        if st.session_state.parsed_job_info and st.session_state.parsed_resume_info:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Generate Interview Questions", type="primary", use_container_width=True):
                    with st.spinner("Generating personalized interview questions..."):
                        questions = generate_interview_questions(
                            st.session_state.parsed_job_info,
                            st.session_state.parsed_resume_info
                        )
                        
                        if questions:
                            st.session_state.generated_questions = questions
                            st.success("Interview questions generated successfully!")
                        else:
                            st.error("Failed to generate questions. Please try again.")
    
    # Display generated questions
    if st.session_state.get('generated_questions') is not None and st.session_state.generated_questions:
        st.divider()
        render_questions_display()
        render_download_section()
        
        # Reset button
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Start Over", use_container_width=True):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    if key.startswith(('job_', 'resume_', 'parsed_', 'generated_', 'parsing_', 'compatibility_')):
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()