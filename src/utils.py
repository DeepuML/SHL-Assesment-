"""
Utility Functions
=================
Helper functions used across the SHL recommendation system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_json(filepath: str) -> Any:
    """
    Load JSON data from file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON to: {filepath}")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length (L2 normalization).
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def batch_process(items: List[Any], batch_size: int = 32):
    """
    Generator for batch processing.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def setup_logging(log_file: str = None, level: str = "INFO"):
    """
    Configure logging for the application.
    
    Args:
        log_file: Optional log file path
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def ensure_dir(dirpath: str) -> None:
    """Ensure directory exists."""
    Path(dirpath).mkdir(parents=True, exist_ok=True)
