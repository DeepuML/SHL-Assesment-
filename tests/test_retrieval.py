"""
Test Retrieval Engine
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_retrieval_basic():
    """Test basic retrieval functionality."""
    from src.retrieval_engine import RetrievalEngine

    engine = RetrievalEngine()
    engine.load()

    # Test query
    query = "Python developer"
    results = engine.recommend(query, k=5)

    assert len(results) <= 5
    assert all("assessment_name" in r for r in results)
    assert all("assessment_url" in r for r in results)

    print("Basic retrieval test passed")


if __name__ == "__main__":
    test_retrieval_basic()
