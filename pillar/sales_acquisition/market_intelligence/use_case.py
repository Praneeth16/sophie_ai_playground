import streamlit as st
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

import openai
from tavily import TavilyClient
from pydantic import BaseModel, Field
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Configuration & Models =============
@dataclass
class ResearchConfig:
    """Research agent configuration"""
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "your_tavily_api_key_here")
    model_name: str = "moonshotai/kimi-k2"
    search_depth: str = "advanced"
    max_search_results: int = 8
    max_sections: int = 4
    max_queries_per_section: int = 2
    temperature: float = 0.7
    max_retries: int = 8

class ResearchPhase(Enum):
    """Research workflow phases"""
    IDLE = "idle"
    BRIEF_GENERATION = "brief_generation"
    SECTION_PLANNING = "section_planning"
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    ERROR = "error"

class ResearchBrief(BaseModel):
    """Structured research brief"""
    objective: str = Field(description="Clear research objective")
    key_questions: List[str] = Field(description="Key questions to answer")
    scope: str = Field(description="Research scope and boundaries")
    expected_outcome: str = Field(description="Expected deliverable")

class ResearchSection(BaseModel):
    """Individual research section"""
    title: str = Field(description="Section title")
    description: str = Field(description="What this section covers")
    search_queries: List[str] = Field(description="Search queries for this section")

class ResearchPlan(BaseModel):
    """Complete research plan"""
    sections: List[ResearchSection] = Field(description="Research sections")
    
class SectionContent(BaseModel):
    """Content for a research section"""
    title: str
    content: str
    sources: List[Dict[str, str]]

@dataclass
class ResearchState:
    """Research workflow state management"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: ResearchPhase = ResearchPhase.IDLE
    topic: str = ""
    brief: Optional[ResearchBrief] = None
    plan: Optional[ResearchPlan] = None
    search_results: Dict[str, List[Dict]] = field(default_factory=dict)
    sections: List[SectionContent] = field(default_factory=list)
    final_report: str = ""
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

# ============= API Clients =============
class OpenRouterClient:
    """OpenRouter API client"""
    
    def __init__(self, api_key: str, model_name: str = "moonshotai/kimi-k2"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/deep-research-agent",
                "X-Title": "Deep Research Agent"
            }
        )
    
    def complete(self, prompt: str, system: str = None, temperature: float = 0.7, 
                response_format: type = None, max_retries: int = 3) -> Any:
        """Generate completion with error handling"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(max_retries):
            try:
                if response_format:
                    json_prompt = f"{prompt}\n\nRespond with only valid JSON, no other text or formatting."
                    messages[-1]["content"] = json_prompt
                    
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content.strip()
                    
                    # Clean up JSON formatting
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    return response_format.model_validate_json(content)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature
                    )
                    return response.choices[0].message.content
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"OpenRouter API error: {str(e)}")
                raise e

class TavilySearchClient:
    """Tavily search client"""
    
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)
    
    def search(self, query: str, search_depth: str = "advanced", 
               max_results: int = 8) -> List[Dict]:
        """Execute search"""
        try:
            results = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_raw_content=True,
                include_answer=True
            )
            
            formatted_results = []
            for result in results.get('results', []):
                formatted_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'score': result.get('score', 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# ============= Research Agent =============
class ResearchAgent:
    """Research agent"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.llm = OpenRouterClient(config.openrouter_api_key, config.model_name)
        self.search = TavilySearchClient(config.tavily_api_key)
        
    async def generate_brief(self, topic: str) -> ResearchBrief:
        """Generate research brief"""
        prompt = f"""Create a focused research brief for: {topic}

Return a JSON object with these fields:
- objective: Clear research goal (string)
- key_questions: 3-4 specific questions (array of strings)
- scope: Research boundaries (string)  
- expected_outcome: Expected deliverable (string)

Topic: {topic}"""
        
        system = "You are a market research strategist. Return only valid JSON with simple strings and arrays."
        
        return self.llm.complete(prompt, system, temperature=0.3, 
                                response_format=ResearchBrief)
    
    async def create_research_plan(self, brief: ResearchBrief) -> ResearchPlan:
        """Create research plan"""
        prompt = f"""Create a research plan with {self.config.max_sections} sections.

Objective: {brief.objective}
Questions: {json.dumps(brief.key_questions)}

Return JSON with:
{{
  "sections": [
    {{
      "title": "Section name",
      "description": "What it covers", 
      "search_queries": ["Query 1", "Query 2"]
    }}
  ]
}}"""
        
        system = "You are a business intelligence researcher. Return only valid JSON."
        
        return self.llm.complete(prompt, system, temperature=0.5, 
                                response_format=ResearchPlan)
    
    async def research_section(self, section: ResearchSection, 
                             brief: ResearchBrief) -> Tuple[SectionContent, Dict[str, List[Dict]]]:
        """Research individual section"""
        all_results = {}
        combined_content = []
        all_sources = []
        
        # Execute searches
        for query in section.search_queries[:self.config.max_queries_per_section]:
            results = self.search.search(
                query, 
                self.config.search_depth,
                self.config.max_search_results
            )
            all_results[query] = results
            
            for result in results[:4]:
                combined_content.append(f"Source: {result['title']}\n{result['content']}")
                all_sources.append({
                    'title': result['title'],
                    'url': result['url'],
                    'query': query
                })
        
        # Synthesize content
        prompt = f"""Write a comprehensive section for a market intelligence report.

Section: {section.title}
Description: {section.description}
Research Objective: {brief.objective}

Based on these search results:
{chr(10).join(combined_content[:8])}

Write 2-3 detailed paragraphs with specific insights and business implications."""
        
        system = "You are a market intelligence analyst. Synthesize information into actionable insights."
        
        content = self.llm.complete(prompt, system, temperature=0.7)
        
        return SectionContent(
            title=section.title,
            content=content,
            sources=all_sources[:8]
        ), all_results
    
    async def generate_final_report(self, brief: ResearchBrief, 
                                   sections: List[SectionContent]) -> str:
        """Generate final report"""
        sections_text = "\n\n".join([
            f"## {s.title}\n\n{s.content}" for s in sections
        ])
        
        prompt = f"""Create a professional market intelligence report.

Research Objective: {brief.objective}

Sections Content:
{sections_text}

Write an Executive Summary and Strategic Recommendations to complete the report."""
        
        system = "You are a senior business analyst. Create executive-level reports."
        
        report_parts = self.llm.complete(prompt, system, temperature=0.6)
        
        # Compile final report
        report = f"# Market Intelligence Report\n## {brief.objective}\n\n"
        report += f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        report += report_parts + "\n\n"
        report += sections_text + "\n\n"
        
        # Add sources
        report += "\n## Sources\n\n"
        seen_urls = set()
        for section in sections:
            for source in section.sources:
                if source['url'] not in seen_urls:
                    seen_urls.add(source['url'])
                    report += f"- [{source['title']}]({source['url']})\n"
        
        return report

# ============= UI Functions =============
def init_session_state():
    """Initialize session state"""
    if 'research_state' not in st.session_state:
        st.session_state.research_state = ResearchState()
    if 'config' not in st.session_state:
        st.session_state.config = ResearchConfig()

def get_api_keys():
    """Get API keys from environment"""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    return openrouter_key, tavily_key

def run_research(topic: str, config: ResearchConfig):
    """Run research with simple progress tracking"""
    try:
        # Simple progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize state
        state = st.session_state.research_state
        state.topic = topic
        state.start_time = time.time()
        state.phase = ResearchPhase.BRIEF_GENERATION
        
        agent = ResearchAgent(config)
        
        # Step 1: Generate brief
        status_text.text("🎯 Generating research brief...")
        progress_bar.progress(20)
        
        state.brief = asyncio.run(agent.generate_brief(topic))
        state.phase = ResearchPhase.SECTION_PLANNING
        
        # Step 2: Create plan
        status_text.text("📋 Creating research plan...")
        progress_bar.progress(40)
        
        state.plan = asyncio.run(agent.create_research_plan(state.brief))
        state.phase = ResearchPhase.RESEARCH
        
        # Step 3: Research sections
        total_sections = len(state.plan.sections)
        for i, section in enumerate(state.plan.sections):
            status_text.text(f"🔍 Researching: {section.title}")
            section_progress = 40 + (40 * (i + 1) / total_sections)
            progress_bar.progress(int(section_progress))
            
            section_content, search_results = asyncio.run(
                agent.research_section(section, state.brief)
            )
            state.sections.append(section_content)
            state.search_results.update(search_results)
        
        # Step 4: Generate report
        state.phase = ResearchPhase.SYNTHESIS
        status_text.text("📝 Generating final report...")
        progress_bar.progress(90)
        
        state.final_report = asyncio.run(
            agent.generate_final_report(state.brief, state.sections)
        )
        
        # Complete
        state.phase = ResearchPhase.COMPLETE
        state.end_time = time.time()
        
        status_text.text("✅ Research complete!")
        progress_bar.progress(100)
        
        # Update session state
        st.session_state.research_state = state
        
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Show success message
        duration = round(state.end_time - state.start_time, 1)
        st.success(f"🎯 Research completed in {duration} seconds! Your comprehensive market intelligence report is ready below.")
        
    except Exception as e:
        state.phase = ResearchPhase.ERROR
        state.error = str(e)
        st.session_state.research_state = state
        
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"❌ Research failed: {str(e)}")
        
        with st.expander("Troubleshooting Tips"):
            st.markdown("""
            **Common solutions:**
            - Check your API keys in .env file
            - Wait a few moments and try again
            - Ensure you have API credits
            - Try a simpler research topic
            """)

def render_hero_section():
    """Render hero section"""
    st.markdown("""
    <style>
    .hero-container {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: linear-gradient(135deg, rgba(244, 228, 217, 0.3) 0%, rgba(240, 213, 196, 0.2) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 1.5rem;
        color: #374151;
        background: linear-gradient(135deg, #374151 0%, #d4896a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: #4a5568;
        line-height: 1.5;
        max-width: 800px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
        flex-wrap: wrap;
    }
    
    .hero-feature {
        background: rgba(212, 137, 106, 0.1);
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        color: #c67651;
        font-weight: 600;
        border: 1px solid rgba(212, 137, 106, 0.2);
    }
    </style>
    
    <div class="hero-container">
        <div class="hero-title">Deep Market Intelligence</div>
        <div class="hero-subtitle">
            AI-powered research agent that generates comprehensive market intelligence reports 
            with competitor analysis, trend insights, and strategic recommendations
        </div>
        <div class="hero-features">
            <div class="hero-feature">Comprehensive Analysis</div>
            <div class="hero-feature">Strategic Insights</div>
            <div class="hero-feature">AI-Powered Speed</div>
            <div class="hero-feature">Executive Reports</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def check_api_status(config: ResearchConfig) -> Dict[str, bool]:
    """Check API configuration"""
    status = {
        'openrouter': bool(config.openrouter_api_key and config.openrouter_api_key != "your_openrouter_api_key_here"),
        'tavily': bool(config.tavily_api_key and config.tavily_api_key != "your_tavily_api_key_here"),
    }
    status['all_good'] = all(status.values())
    return status

def render_research_interface():
    """Render main research interface"""
    state = st.session_state.research_state
    
    # Always show input interface
    render_research_input()
    
    # Show results if research is complete
    if (state.phase == ResearchPhase.COMPLETE and state.final_report):
        st.divider()
        st.markdown("## Research Results")
        render_complete_state(state)
        
        # New research button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Start New Research", type="secondary", use_container_width=True):
                st.session_state.research_state = ResearchState()

def render_research_input():
    """Render research input interface"""
    st.markdown("## Enter Your Research Topic")
    
    topic = st.text_area(
        "What would you like to research?",
        placeholder="Enter a detailed research question, e.g., 'Analyze the AI-powered recruitment tools market including key players, growth trends, competitive landscape, and adoption barriers'",
        height=150,
        help="Be specific about your research needs. Include industry, scope, and key questions you want answered."
    )
    
    # Configuration
    with st.expander("Advanced Settings", expanded=False):
        openrouter_key, tavily_key = get_api_keys()
        config = st.session_state.config
        config.tavily_api_key = tavily_key
        config.openrouter_api_key = openrouter_key
        
        config.max_sections = st.slider("Research Sections", 2, 6, 4)
        config.search_depth = st.selectbox("Search Depth", ["basic", "advanced"], index=1)
        config.max_search_results = st.slider("Results per Search", 5, 15, 8)
    
    # Start research button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        api_status = check_api_status(st.session_state.config)
        button_disabled = not topic or not api_status['all_good']
        
        if st.button("Generate Market Intelligence Report", type="primary", disabled=button_disabled, use_container_width=True):
            if not api_status['all_good']:
                st.error("Please check your API key configuration in the .env file")
            else:
                run_research(topic, st.session_state.config)
    
    if not api_status['all_good']:
        st.warning("Please ensure your API keys are configured in the .env file")

def render_complete_state(state: ResearchState):
    """Render completion state with results"""
    # Summary metrics
    if state.end_time and state.start_time:
        duration = state.end_time - state.start_time
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("Sections", len(state.sections))
        with col3:
            word_count = len(state.final_report.split())
            st.metric("Words", f"{word_count:,}")
    
    # Report tabs
    tab1, tab2, tab3 = st.tabs(["Final Report", "Research Sections", "Sources"])
    
    with tab1:
        if state.final_report:
            st.markdown(state.final_report)
        else:
            st.error("No report content available.")
        
        # Download button
        st.download_button(
            label="📄 Download Report (Markdown)",
            data=state.final_report if state.final_report else "",
            file_name=f"market_intelligence_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with tab2:
        st.markdown("### Research Sections")
        for i, section in enumerate(state.sections, 1):
            with st.expander(f"Section {i}: {section.title}", expanded=False):
                st.markdown(section.content)
                if section.sources:
                    st.markdown("**Sources:**")
                    for source in section.sources[:3]:
                        st.markdown(f"- [{source['title']}]({source['url']})")
    
    with tab3:
        st.markdown("### All Research Sources")
        seen_urls = set()
        all_sources = []
        
        for section in state.sections:
            for source in section.sources:
                if source['url'] not in seen_urls:
                    seen_urls.add(source['url'])
                    all_sources.append(source)
        
        for i, source in enumerate(all_sources, 1):
            st.markdown(f"{i}. [{source['title']}]({source['url']})")

# ============= Main App =============
def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Deep Market Intelligence",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    init_session_state()
    render_hero_section()
    st.divider()
    render_research_interface()

if __name__ == "__main__":
    main()