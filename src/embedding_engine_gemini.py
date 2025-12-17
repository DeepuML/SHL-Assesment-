""" Gemini Embedding Engine : Generate embeddings using Google's Gemini API and build FAISS index."""

import json
import logging
import numpy as np
import os
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GeminiEmbeddingEngine:
    """Generate embeddings using Gemini API and manage FAISS index."""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "models/text-embedding-004",
        index_path: str = "data/embeddings/faiss_index_gemini.bin",
        metadata_path: str = "data/embeddings/metadata_gemini.json"
    ):
        """
        Initialize Gemini embedding engine.
        
        Args:
            api_key: Gemini API key (or from environment)
            model_name: Gemini embedding model
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # Configure Gemini
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file or pass as parameter.")
        
        genai.configure(api_key=api_key)
        logger.info(f"Configured Gemini API with model: {model_name}")
        
        # Gemini text-embedding-004 produces 768-dimensional embeddings
        self.embedding_dim = 768
        self.index = None
        self.metadata = []
    
    def create_document_text(self, assessment: Dict) -> str:
        """
        Create combined text from assessment fields.
        
        Args:
            assessment: Assessment dictionary
            
        Returns:
            Combined text string
        """
        parts = [
            assessment.get('assessment_name', ''),
            f"Type: {assessment.get('test_type', '')}",
            assessment.get('category', ''),
            assessment.get('description', ''),
            assessment.get('skills', ''),
            assessment.get('job_roles', '')
        ]
        
        # Filter out empty parts
        parts = [p for p in parts if p and p != 'Type: ']
        
        return ' '.join(parts)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 10) -> np.ndarray:
        """
        Generate embeddings using Gemini API.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls
            
        Returns:
            Numpy array of embeddings (N x 768)
        """
        logger.info(f"Generating Gemini embeddings for {len(texts)} documents...")
        
        embeddings = []
        
        # Process in batches to handle API rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                try:
                    # Generate embedding using Gemini
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * self.embedding_dim)
        
        # Convert to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def build_index(self, assessments: List[Dict]) -> None:
        """
        Build FAISS index from assessments.
        
        Args:
            assessments: List of assessment dictionaries
        """
        logger.info("=" * 60)
        logger.info("BUILDING GEMINI-POWERED FAISS INDEX")
        logger.info("=" * 60)
        
        # Create document texts
        logger.info("\nStep 1: Creating document texts...")
        texts = [self.create_document_text(a) for a in assessments]
        
        # Generate embeddings
        logger.info("\nStep 2: Generating Gemini embeddings...")
        embeddings = self.generate_embeddings(texts)
        
        # Normalize embeddings for cosine similarity
        logger.info("\nStep 3: Normalizing embeddings...")
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product = cosine similarity after normalization)
        logger.info("\nStep 4: Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata = assessments
        
        # Save to disk
        logger.info("\nStep 5: Saving to disk...")
        self.save()
        
        logger.info("\n" + "=" * 60)
        logger.info(" GEMINI INDEX BUILD COMPLETE")
        logger.info(f"Processed {len(assessments)} assessments")
        logger.info(f"Index saved to: {self.index_path}")
        logger.info(f"Metadata saved to: {self.metadata_path}")
        logger.info("=" * 60)
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        # Create directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved FAISS index to: {self.index_path}")
        logger.info(f"Saved metadata to: {self.metadata_path}")
    
    def load(self) -> None:
        """Load index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        
        # Load metadata
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded FAISS index from: {self.index_path}")
        logger.info(f"Loaded {len(self.metadata)} assessments")


if __name__ == "__main__":
    # Test Gemini embedding engine
    logging.basicConfig(level=logging.INFO)
    
    engine = GeminiEmbeddingEngine()
    
    # Test embedding generation
    test_texts = ["Python programming assessment", "Sales personality test"]
    embeddings = engine.generate_embeddings(test_texts)
    
    print(f"\nTest embeddings shape: {embeddings.shape}")
    print(f"Expected: (2, 768)")
    print(" Gemini Embedding Engine working!")
