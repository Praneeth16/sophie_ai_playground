import streamlit as st

# Sales Acquisition use cases data
use_cases_data = [
    {"name": "Smart Document Intelligence for Proposals", "description": "AI-powered document analysis and intelligent Q&A for client proposals, contracts, and recruitment materials"},
    {"name": "Personality-Driven Client Presentations", "description": "Dynamic sales presentations automatically adapted to hiring manager personality types and company culture"},
    {"name": "Real-Time Industry Intelligence Dashboard", "description": "Live workforce trends, salary data, and company news to fuel informed client conversations"},
    {"name": "Decision Maker Discovery Engine", "description": "AI-driven identification of key hiring stakeholders and influencers within target organizations"},
    {"name": "CRM Data Optimization Suite", "description": "Automated client data cleaning, standardization, and enrichment for recruitment CRM systems"},
    {"name": "Competitive Market Intelligence Reports", "description": "Comprehensive talent market analysis and competitive recruitment intelligence for strategic positioning"}
]

def render_pillar_page():
    """Render Sales Acquisition pillar page with use cases in card format"""
    st.markdown("# Sales Acquisition")
    st.markdown("**AI-powered tools and solutions for sales acquisition**")
    st.write("")
    
    # Display tools in card grid format
    for i in range(0, len(use_cases_data), 2):
        cols = st.columns(2, gap="large")
        
        for j, tool in enumerate(use_cases_data[i:i+2]):
            if j < len(cols):
                with cols[j]:
                    # Create card-style container
                    with st.container(border=True):
                        st.markdown(f"### {tool['name']}")
                        st.markdown(tool['description'])
                        
                        st.write("")
                        st.markdown("*Click on the tool name in the navigation dropdown above to explore this tool in detail.*")

def render_use_case_page(use_case_name):
    """Render an individual use case page for Sales Acquisition"""
    st.markdown(f"# {use_case_name}")
    
    # Back button to pillar page
    col_back, col_spacer = st.columns([2, 8])
    with col_back:
        if st.button("← Back to Sales Acquisition", use_container_width=True):
            # Clear use case selection to go back to pillar page
            if 'selected_use_case' in st.session_state:
                del st.session_state.selected_use_case
            if 'selected_pillar' in st.session_state:
                del st.session_state.selected_pillar
            st.rerun()
    
    st.divider()
    
    # Find the use case data
    selected_tool = next((tool for tool in use_cases_data if tool['name'] == use_case_name), None)
    
    if selected_tool:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## Overview")
            st.markdown(selected_tool['description'])
            
            st.markdown("## Key Features")
            st.markdown("""
            - 🚀 **AI-Powered**: Advanced machine learning algorithms
            - ⚡ **Real-time Processing**: Instant results and updates
            - 🔧 **Easy Integration**: Seamless workflow integration
            - 📊 **Analytics Dashboard**: Comprehensive performance metrics
            - 🛡️ **Enterprise Security**: Bank-level data protection
            """)
            
            st.markdown("## Benefits")
            st.markdown("""
            - **Increased Efficiency**: Save 75% of manual processing time
            - **Improved Accuracy**: AI-driven precision reduces errors by 90%
            - **Better Outcomes**: Enhanced results and user satisfaction
            - **Cost Savings**: Reduce operational costs significantly
            """)
            
        with col2:
            st.markdown("## Quick Actions")
            
            with st.container(border=True):
                if st.button("🚀 Try Demo", use_container_width=True):
                    st.success("Demo launched! Experience the power of AI-driven recruitment.")
                
                if st.button("📞 Schedule Call", use_container_width=True):
                    st.info("Scheduling a call with our specialists...")
                
                if st.button("📚 Documentation", use_container_width=True):
                    st.info("Opening comprehensive documentation...")
                
                if st.button("💬 Contact Support", use_container_width=True):
                    st.info("Connecting you with our support team...")
            
            st.markdown("## Related Tools")
            # Show other tools from the same pillar
            other_tools = [tool for tool in use_cases_data if tool['name'] != use_case_name]
            for tool in other_tools[:3]:  # Show max 3 related tools
                with st.container(border=True):
                    st.markdown(f"**{tool['name']}**")
                    st.markdown(tool['description'][:100] + "..." if len(tool['description']) > 100 else tool['description'])
    else:
        st.error(f"Use case '{use_case_name}' not found in Sales Acquisition.")

def sales_acquisition_page():
    """Sales Acquisition pillar page"""
    render_pillar_page() 