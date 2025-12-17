# SHL Assessment Recommendation System

A production-ready GenAI-powered system that recommends relevant SHL assessments based on natural language queries, job descriptions, or job posting URLs.

## Project Overview

This system uses semantic search and intelligent balancing to recommend 5-10 SHL individual assessments from a catalog of 377+ tests. It combines embedding-based retrieval with test-type balancing to ensure diverse, relevant recommendations spanning both technical (Type K) and behavioral (Type P) assessments.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

### Key Components:

- **Web Scraper**: Extracts 377+ individual assessments from SHL product catalog
- **Embedding Engine**: Converts assessments to semantic vectors using SentenceTransformers
- **FAISS Index**: Enables fast similarity search
- **Retrieval Engine**: Semantic search with test-type balancing
- **FastAPI Backend**: RESTful API with `/health` and `/recommend` endpoints
- **Streamlit Frontend**: User-friendly web interface
- **Evaluation Module**: Computes Mean Recall@10 on labeled data

## Quick Start

### Option 1: Docker (Recommended)

**Prerequisites**: Docker and Docker Compose installed

```bash
# Quick start with Docker
docker-compose up --build

# Or use the startup script
# Windows:
docker-start.bat

# Linux/Mac:
chmod +x docker-start.sh
./docker-start.sh
```

Access services:

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

See [DOCKER_CICD.md](DOCKER_CICD.md) for detailed Docker documentation.

### Option 2: Manual Setup

#### 1. Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

#### 2. Data Collection

```bash
# Scrape SHL product catalog
python scripts/run_scraper.py
```

#### 3. Build Index

```bash
# Generate embeddings and create FAISS index
python scripts/build_index.py
```

### 4. Run Evaluation

```bash
# Evaluate on labeled train set
python scripts/evaluate_model.py
```

#### 5. Start API Server

```bash
# Start FastAPI backend
uvicorn api.app:app --reload --port 8000
```

#### 6. Launch Web Interface

```bash
# Start Streamlit frontend (in new terminal)
streamlit run frontend/streamlit_app.py
```

## Docker & CI/CD

### Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild
docker-compose up --build

# Using Makefile (if available)
make build      # Build images
make up         # Start services
make logs       # View logs
make test       # Run tests
make clean      # Clean up
```

### CI/CD Pipeline

The project includes a complete GitHub Actions CI/CD pipeline that runs on every push:

- **Lint**: Code quality checks (flake8, black, isort)
- **Test**: Automated testing with pytest
- **Build**: Docker image creation and push to registry
- **Security**: Vulnerability scanning with Trivy
- **Deploy**: Automatic deployment to production (main branch)

See [DOCKER_CICD.md](DOCKER_CICD.md) for complete documentation.

## API Usage

### Health Check

```bash
GET http://localhost:8000/health

Response: {"status": "ok"}
```

### Get Recommendations

```bash
POST http://localhost:8000/recommend
Content-Type: application/json

{
  "query": "Looking for a senior Python developer with strong leadership skills"
}

Response:
{
  "recommendations": [
    {
      "assessment_name": "Python Programming Assessment",
      "assessment_url": "https://www.shl.com/solutions/products/..."
    },
    ...
  ]
}
```

## Evaluation

The system is evaluated using **Mean Recall@10** on a labeled training set:

```
Recall@10 = |Recommended ∩ Relevant| / |Relevant|
Mean Recall@10 = Average across all queries
```

### Expected Performance:

- **Baseline** (pure semantic search): ~0.35-0.45
- **With balancing**: ~0.45-0.55
- **With query enrichment**: ~0.50-0.60
- **With re-ranking**: ~0.55-0.65

## Project Structure

```
SHL Assessment/
├── data/                           # Data storage
│   ├── raw/                        # Scraped data
│   ├── processed/                  # Cleaned data
│   ├── embeddings/                 # FAISS index & metadata
│   ├── train/                      # Labeled training data
│   └── test/                       # Test data
├── src/                            # Source code
│   ├── scraper.py                  # Web scraping
│   ├── data_processor.py           # Data cleaning
│   ├── embedding_engine.py         # Embeddings + FAISS
│   ├── retrieval_engine.py         # Retrieval + balancing
│   ├── evaluation.py               # Evaluation metrics
│   └── utils.py                    # Helper functions
├── api/                            # FastAPI backend
│   └── app.py
├── frontend/                       # Streamlit UI
│   └── streamlit_app.py
├── scripts/                        # Execution scripts
│   ├── run_scraper.py
│   ├── build_index.py
│   ├── evaluate_model.py
│   └── generate_predictions.py
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks
├── requirements.txt                # Python dependencies
├── ARCHITECTURE.md                 # Detailed design doc
└── README.md                       # This file
```

## Key Features

### 1. Semantic Search

Uses SentenceTransformers to understand query intent beyond keywords.

### 2. Test-Type Balancing

Ensures balanced recommendations when queries span multiple skill domains:

- Technical queries → Prioritize Type K (Knowledge/Skills)
- Behavioral queries → Prioritize Type P (Personality/Behavior)
- Mixed queries → Balanced distribution

### 3. Multiple Input Formats

- Natural language queries
- Full job descriptions (copy-paste)
- Job posting URLs (automatic extraction)

### 4. Evaluation Framework

Rigorous evaluation with Mean Recall@10 to measure recommendation quality.

## Technology Stack

- **Language**: Python 3.9+
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Web Scraping**: BeautifulSoup4 + Selenium
- **Data Processing**: Pandas, NumPy

## Requirements

- Python 3.9+
- 4GB RAM minimum (for FAISS index)
- Internet connection (for scraping and embeddings)

## Testing

```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_retrieval.py -v
```

## Generating Predictions

```bash
# Generate predictions for test set
python scripts/generate_predictions.py

# Output: data/predictions/submission.csv
```

Submission format:

```csv
query_id,recommended_assessments
1,"url1,url2,url3,..."
2,"url1,url2,url3,..."
```

## Design Decisions

### Why SentenceTransformers?

- Free and fast (no API costs)
- Local inference (privacy + reliability)
- High-quality embeddings for semantic search

### Why FAISS?

- Extremely fast for <10K vectors
- No external dependencies
- Deterministic and reproducible

### Why Test-Type Balancing?

Pure semantic search can bias toward one assessment type. Balancing ensures comprehensive hiring assessment coverage.

## Performance Optimization

- **Batch embedding generation** for faster indexing
- **FAISS IndexFlatIP** for accurate cosine similarity
- **Caching** of embeddings and metadata
- **Async API** for concurrent requests

## Troubleshooting

### Scraper Issues

- Ensure Chrome/Firefox browser is installed
- Check internet connection
- Verify SHL website structure hasn't changed

### FAISS Errors

- Ensure numpy arrays are float32
- Normalize embeddings before indexing
- Check dimension consistency

### Low Recall@10

- Verify data quality (377+ assessments)
- Check embedding model
- Tune balancing parameters
- Add query enrichment

## License

This project is created for SHL assessment purposes.

## Author

**Senior AI/ML Engineer**  
Date: December 16, 2025

---

**Status**: Architecture Complete  
**Next**: Implementation Phase
