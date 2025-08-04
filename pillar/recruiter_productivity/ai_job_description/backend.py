from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS
import asyncio
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Job Description Generator API",
    description="Generate compelling job descriptions using AI agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

# --- Request/Response Models ---
class JobDescriptionRequest(BaseModel):
    job_title: str = Field(..., description="The job title for the position", example="Senior Backend Engineer")
    company_name: str = Field(..., description="Company name for research and personalization", example="TechCorp Inc.")
    recruiter_summary: str = Field(..., description="Hiring manager brief with role requirements", 
                                  example="We need a senior backend engineer with 5+ years experience in Python and distributed systems...")
    salary_info: Optional[str] = Field(None, description="Salary range or compensation details", 
                                      example="$120,000 - $150,000 + equity + benefits")
    remote_policy: Optional[str] = Field(None, description="Work arrangement policy", 
                                        example="Remote-first, Hybrid (3 days office), On-site required")
    location: Optional[str] = Field(None, description="Job location", example="San Francisco, CA")
    model: Optional[str] = Field("moonshotai/kimi-k2", description="AI model to use for generation")
    temperature: Optional[float] = Field(0.7, description="Temperature for AI generation", ge=0.0, le=2.0)

class JobDescriptionResponse(BaseModel):
    success: bool = Field(..., description="Whether the generation was successful")
    job_description: Optional[str] = Field(None, description="Generated job description in markdown format")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    metadata: Optional[dict] = Field(None, description="Additional metadata about the generation process")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")
    timestamp: str = Field(..., description="Current timestamp")

# --- Helper Functions ---

def get_openrouter_client():
    """Initialize OpenRouter client using OpenAI SDK"""
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
        return None
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

async def call_openrouter_api(model: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2000):
    """Makes a call to the OpenRouter API using OpenAI client."""
    client = get_openrouter_client()
    if client is None:
        raise HTTPException(status_code=500, detail="OpenRouter API key is not configured")

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://hr-copilot.api",
                "X-Title": "HR CoPilot - AI Job Description Generator API",
            },
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Request Error: {str(e)}")

@lru_cache(maxsize=100)
def search_company_info_cached(company_name: str):
    """Cached company information lookup"""
    try:
        with DDGS() as ddgs:
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
                return "No specific company information found."
            
            combined_info = " ".join([r['body'] for r in all_results])
            return combined_info[:2000] if len(combined_info) > 2000 else combined_info
            
    except Exception as e:
        return f"Company research unavailable: {str(e)}"

async def experience_extraction_agent(model: str, job_title: str, recruiter_summary: str):
    """Agent 1: Extract experience requirements and analyze role level."""
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
    
    return await call_openrouter_api(model, prompt, temperature=0.1)

async def market_research_agent(model: str, job_title: str, company_info: str):
    """Agent 2: Market research and competitive analysis."""
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
    
    return await call_openrouter_api(model, prompt, temperature=0.3)

async def job_description_architect(model: str, job_title: str, experience_analysis: str, company_name: str, 
                                  company_info: str, recruiter_summary: str, market_research: str, 
                                  salary_info: Optional[str], remote_policy: Optional[str], location: Optional[str]):
    """Agent 3: Master job description architect."""
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
    
    return await call_openrouter_api(model, prompt, temperature=0.4)

async def optimization_agent(model: str, job_description_draft: str, job_title: str):
    """Agent 4: Final optimization for ATS, SEO, and candidate appeal."""
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
    
    return await call_openrouter_api(model, prompt, temperature=0.2)

def clean_markdown_content(content: str) -> str:
    """Clean and validate markdown content for proper rendering"""
    if not content:
        return "No content generated."
    
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

# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=str(time.time())
    )

@app.post("/generate-job-description", response_model=JobDescriptionResponse)
async def generate_job_description(request: JobDescriptionRequest):
    """Generate a job description using AI agents"""
    
    # Validate API key
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
        raise HTTPException(
            status_code=500, 
            detail="OpenRouter API key is not configured. Please set OPENROUTER_API_KEY environment variable."
        )
    
    try:
        start_time = time.time()
        
        # Agent 1: Experience Extraction
        experience_analysis = await experience_extraction_agent(
            request.model, request.job_title, request.recruiter_summary
        )
        
        # Agent 2: Company Research and Market Analysis
        company_info = search_company_info_cached(request.company_name)
        market_research = await market_research_agent(
            request.model, request.job_title, company_info
        )
        
        # Agent 3: Job Description Architecture
        initial_draft = await job_description_architect(
            request.model, request.job_title, experience_analysis, request.company_name,
            company_info, request.recruiter_summary, market_research,
            request.salary_info, request.remote_policy, request.location
        )
        
        # Agent 4: Final Optimization
        final_job_description = await optimization_agent(
            request.model, initial_draft, request.job_title
        )
        
        # Clean the markdown content
        final_job_description = clean_markdown_content(final_job_description)
        
        processing_time = time.time() - start_time
        
        return JobDescriptionResponse(
            success=True,
            job_description=final_job_description,
            metadata={
                "processing_time_seconds": round(processing_time, 2),
                "model_used": request.model,
                "company_research_available": len(company_info) > 50,
                "timestamp": str(time.time())
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        return JobDescriptionResponse(
            success=False,
            error=f"An error occurred during generation: {str(e)}"
        )

@app.get("/models")
async def get_available_models():
    """Get list of available AI models"""
    return {
        "models": [
            "moonshotai/kimi-k2",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-8b-instruct"
        ],
        "default": "moonshotai/kimi-k2"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)