"""Streamlit Frontend : Web interface for SHL Assessment Recommendation System."""

import sys
from pathlib import Path

import requests
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure page
st.set_page_config(page_title="SHL Assessment Recommender", page_icon="", layout="wide")

# API endpoint
API_URL = "http://localhost:8000"


def get_recommendations(query: str):
    """Get recommendations from API.
    Args:
        query: Search query
    Returns:
        List of recommendations or None if error
    """
    try:
        response = requests.post(
            f"{API_URL}/recommend", json={"query": query}, timeout=30
        )
        response.raise_for_status()
        return response.json()["recommendations"]
    except requests.exceptions.ConnectionError:
        st.error(
            " Cannot connect to API. Make sure the backend is running on port 8000."
        )
        return None
    except Exception as e:
        st.error(f" Error: {str(e)}")
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
        st.header("ℹ About")
        st.markdown(
            """This system uses **semantic search** and **AI-powered balancing** to recommend 
        relevant SHL assessments.
        
        ### Features:
        -  Semantic understanding of job requirements
        -  Balanced recommendations (technical + behavioral)
        -  Fast retrieval from 377+ assessments
         ### Powered by:
        - SentenceTransformers
        - FAISS Vector Search
        - FastAPI
        """
        )

        st.divider()

        # Health check
        if st.button("Check API Status"):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                if response.status_code == 200:
                    st.success(" API is running")
                else:
                    st.error(" API returned error")
            except:
                st.error(" API is not accessible")


if __name__ == "__main__":
    main()
