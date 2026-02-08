"""
Streamlit UI for Autonomous Insight Engine.

Interactive interface for uploading CSV, monitoring progress, and viewing insights.
"""
import streamlit as st
import polars as pl
from pathlib import Path
import time
from datetime import datetime

from main import AutonomousInsightEngine
from utils.helpers import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Autonomous Insight Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .sub-header {
        font-size: 1.2em;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .cfo-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .coo-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .cmo-card { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    
    .metric-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'report' not in st.session_state:
    st.session_state.report = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🧠 Autonomous Insight Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Data Analysis & Insight Generation</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.markdown("""
        This system replicates human-grade analytical insights using:
        
        - **LLM #1 (Groq)**: Context inference & hypothesis generation
        - **LLM #2 (HuggingFace)**: Insight narration
        - **Statistical Analysis**: All computations done by code, not LLM
        - **Self-Healing**: Automatic error correction
        - **Adversarial Validation**: Rejects weak insights
        """)
        
        st.header("🔧 Features")
        st.markdown("""
        - ✅ Automatic data cleaning
        - ✅ Semantic profiling
        - ✅ CFO/COO/CMO perspectives
        - ✅ Time series forecasting
        - ✅ Causal analysis
        - ✅ Anomaly detection
        - ✅ PDF report generation
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results", "📥 Download"])
    
    with tab1:
        st.header("Upload Your CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload any CSV file with tabular data"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.uploaded_file_path = str(file_path)
            
            # Show preview
            st.subheader("📋 Data Preview")
            try:
                df_preview = pl.read_csv(str(file_path))
                st.dataframe(df_preview.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{len(df_preview):,}")
                with col2:
                    st.metric("Columns", f"{len(df_preview.columns):,}")
                with col3:
                    st.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
                
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
            
            st.divider()
            
            # Run analysis button
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                run_analysis(str(file_path), uploaded_file.name)
    
    with tab2:
        st.header("Analysis Results")
        
        if st.session_state.analysis_complete and st.session_state.report:
            display_results(st.session_state.report)
        else:
            st.info("👈 Upload a CSV file and run analysis to see results here")
    
    with tab3:
        st.header("Download Report")
        
        if st.session_state.analysis_complete:
            # Find most recent report
            reports_dir = Path("reports")
            if reports_dir.exists():
                html_files = list(reports_dir.glob("*.html"))
                pdf_files = list(reports_dir.glob("*.pdf"))
                
                if html_files:
                    latest_html = max(html_files, key=lambda p: p.stat().st_mtime)
                    
                    with open(latest_html, "rb") as f:
                        st.download_button(
                            label="📄 Download HTML Report",
                            data=f,
                            file_name=latest_html.name,
                            mime="text/html",
                            use_container_width=True
                        )
                
                if pdf_files:
                    latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
                    
                    with open(latest_pdf, "rb") as f:
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=f,
                            file_name=latest_pdf.name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                
                st.success("✅ Reports ready for download!")
            else:
                st.warning("No reports found")
        else:
            st.info("Complete an analysis first to download reports")


def run_analysis(csv_path: str, dataset_name: str):
    """Run the analysis pipeline with progress tracking."""
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Analysis in Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        phase_info = st.empty()
        
        try:
            # Initialize engine
            engine = AutonomousInsightEngine()
            
            # Create a placeholder for live updates
            start_time = time.time()
            
            # Run pipeline (this will take some time)
            with st.spinner("Running autonomous analysis..."):
                report = engine.run_pipeline(csv_path, dataset_name)
            
            # Complete
            progress_bar.progress(100)
            status_text.success("✅ Analysis Complete!")
            
            elapsed_time = time.time() - start_time
            phase_info.info(f"Total time: {elapsed_time:.1f} seconds")
            
            # Store results
            st.session_state.analysis_complete = True
            st.session_state.report = report
            
            st.balloons()
            
            # Auto-switch to results tab
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}", exc_info=True)


def display_results(report):
    """Display analysis results."""
    
    # Business Context
    st.subheader("🏢 Business Context")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Industry:** {report.business_context.industry}")
        st.markdown(f"**Business Type:** {report.business_context.business_type}")
    
    with col2:
        st.markdown(f"**Top KPIs:** {', '.join(report.business_context.top_kpis)}")
    
    st.divider()
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    st.info(report.executive_summary)
    
    # Statistics
    st.subheader("📊 Analysis Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Hypotheses Tested",
            report.total_hypotheses,
            help="Total number of hypotheses generated"
        )
    
    with col2:
        st.metric(
            "Validated Insights",
            report.validated_insights,
            help="Insights that passed statistical validation"
        )
    
    with col3:
        st.metric(
            "Rejected",
            report.rejection_count,
            delta=f"-{report.rejection_count}",
            delta_color="inverse",
            help="Insights rejected due to low statistical confidence"
        )
    
    st.divider()
    
    # Insights by Lens
    st.subheader("💡 Key Insights")
    
    # Group insights by lens
    insights_by_lens = {"CFO": [], "COO": [], "CMO": []}
    for insight in report.insights:
        if insight.lens in insights_by_lens:
            insights_by_lens[insight.lens].append(insight)
    
    # Display insights
    for lens, insights in insights_by_lens.items():
        if insights:
            st.markdown(f"### {lens} Perspective ({len(insights)} insights)")
            
            for insight in insights:
                display_insight_card(insight, lens)


def display_insight_card(insight, lens):
    """Display an individual insight card."""
    
    # Confidence badge color
    confidence_colors = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🔴"
    }
    
    badge = confidence_colors.get(insight.confidence, "⚪")
    
    with st.expander(f"{badge} {insight.title}", expanded=True):
        # Key Finding
        st.markdown("**🔍 Key Finding**")
        st.write(insight.key_finding)
        
        # Why It Matters
        st.markdown("**💼 Why It Matters**")
        st.write(insight.why_it_matters)
        
        # Root Cause
        if insight.what_caused_it:
            st.markdown("**🎯 Root Cause**")
            st.write(insight.what_caused_it)
        
        # Recommendation
        st.markdown("**✅ Recommendation**")
        st.success(insight.recommendation)
        
        # Confidence
        st.caption(f"Confidence: {insight.confidence.upper()}")


if __name__ == "__main__":
    main()
