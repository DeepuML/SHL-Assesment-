"""FastAPI Backend : RESTful API for SHL Assessment Recommendation System."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval_engine import RetrievalEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Semantic search API for SHL assessments",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retrieval engine (loaded on startup)
engine = None

class RecommendRequest(BaseModel):
    """Request model for /recommend endpoint."""
    query: str

class Assessment(BaseModel):
    """Assessment model."""
    assessment_name: str
    assessment_url: str

class RecommendResponse(BaseModel):
    """Response model for /recommend endpoint."""
    recommendations: List[Assessment]

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global engine
    logger.info("Loading retrieval engine...")
    
    try:
        engine = RetrievalEngine()
        engine.load()
        logger.info(" Retrieval engine loaded successfully")
    except Exception as e:
        logger.error(f" Failed to load retrieval engine: {e}")
        raise

@app.get("/health")
async def health():
    """ Health check endpoint.
     Returns:
        Health status
    """
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get assessment recommendations for a query. 
    Args:
        request: Request with query text    
    Returns:
        List of recommended assessments
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get recommendations
        recommendations = engine.recommend(request.query, k=10)
        
        # Format response
        assessments = [
            Assessment(
                assessment_name=rec['assessment_name'],
                assessment_url=rec['assessment_url']
            )
            for rec in recommendations
        ]
        
        logger.info(f"Returned {len(assessments)} recommendations for query: {request.query[:50]}...")
        
        return RecommendResponse(recommendations=assessments)
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SHL Assessment Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/recommend": "Get assessment recommendations (POST)",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
