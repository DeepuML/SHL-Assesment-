"""Streamlit App: Standalone SHL Assessment Recommendation System."""

import sys
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure page
st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🎯", layout="wide")

# Load retrieval engine (cached to avoid reloading)
@st.cache_resource
def load_engine():
    """Load retrieval engine once and cache it."""
    from src.retrieval_engine import RetrievalEngine
    return RetrievalEngine()

# Initialize engine
engine = load_engine()


def get_recommendations(query: str):
    """Get recommendations directly from engine.
    Args:
        query: Search query
    Returns:
        List of recommendations or None if error
    """
    try:
        results = engine.recommend(query, k=10)
        # Format results for display
        recommendations = []
        for res in results:
            recommendations.append({
                'assessment_name': res['assessment_name'],
                'assessment_url': res['assessment_url'],
                'similarity_score': res.get('similarity_score', 0)
            })
        return recommendations
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        return None


def main():
    """Main Streamlit app."""

    # Header
    st.title(" SHL Assessment Recommendation System")
    st.markdown(
        """Get personalized SHL assessment recommendations based on your hiring needs.
                Enter a job description, required skills, or a natural language query."""
    )

    st.divider()

    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Natural Language Query", "Job Description (Text)", "Job Posting URL"],
        horizontal=True,
    )

    query = None

    if input_method == "Natural Language Query":
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., Senior Python developer with leadership skills",
            help="Describe the role or skills you're looking for",
        )

    elif input_method == "Job Description (Text)":
        query = st.text_area(
            "Paste job description:",
            height=200,
            placeholder="Paste the full job description here...",
            help="Paste the complete job description text",
        )

    else:  # URL
        url = st.text_input(
            "Enter job posting URL:",
            placeholder="https://example.com/job-posting",
            help="URL to a job posting (URL parsing not yet implemented)",
        )
        if url:
            st.warning(
                "URL parsing not yet implemented. Please use text input for now."
            )

    # Search button
    if st.button("Get Recommendations", type="primary"):
        if query and query.strip():
            with st.spinner("Searching for best assessments..."):
                recommendations = get_recommendations(query.strip())

                if recommendations:
                    st.success(f" Found {len(recommendations)} relevant assessments")
                    st.divider()

                    # Display recommendations
                    st.subheader(" Recommended Assessments")

                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            col1, col2 = st.columns([0.1, 0.9])

                            with col1:
                                st.markdown(f"### {i}")

                            with col2:
                                st.markdown(f"**{rec['assessment_name']}**")
                                st.markdown(
                                    f" [{rec['assessment_url']}]({rec['assessment_url']})"
                                )

                            st.divider()
        else:
            st.warning(" Please enter a query")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """This system uses **semantic search** and **AI-powered balancing** to recommend 
        relevant SHL assessments.
        
        ### Features:
        - 🎯 Semantic understanding of job requirements
        - ⚖️ Balanced recommendations (technical + behavioral)
        - ⚡ Fast retrieval from 300+ assessments
        
        ### Powered by:
        - SentenceTransformers
        - FAISS Vector Search
        - Streamlit
        """
        )

        st.divider()

        # System status
        st.metric("System Status", "🟢 Online")
        st.metric("Total Assessments", len(engine.catalog) if hasattr(engine, 'catalog') else "N/A")


if __name__ == "__main__":
    main()
