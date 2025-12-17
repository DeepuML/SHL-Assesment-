# SHL Assessment Submission Checklist

## 1. Solution Requirements ✓

### ✓ Scraping Pipeline

- **Implementation**: `scraper/playwright_scraper.py`
- **Data Storage**: `data/shl_catalog.json` (270+ products)
- **Parsing**: Extracts title, description, URL, features from SHL website
- **Retrieval**: FAISS vector index with sentence-transformers embeddings

### ✓ Modern LLM/RAG Techniques

- **Embeddings**:
  - Sentence-BERT: `all-MiniLM-L6-v2` (baseline)
  - Google Gemini: `models/text-embedding-004` (enhanced)
- **Retrieval**: Hybrid approach with semantic similarity + BM25 ranking
- **Query Understanding**: Text preprocessing, entity extraction
- **Framework**: LangChain-style RAG pipeline with custom retrieval engine
- **Best Model**: Optimized hyperparameters achieving **21.22% recall@5**

### ✓ Evaluation Methods

- **Metrics**: Recall@5, Precision@5, MRR, NDCG
- **Stages Evaluated**:
  - Baseline model evaluation
  - Gemini embeddings evaluation
  - Enhanced retrieval evaluation
  - Hyperparameter optimization (grid search)
- **Results**: `data/submission_optimized.csv` (best performing model)

---

## 2. Webapp URL

### Local Deployment

```bash
# Start API
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Start Frontend
streamlit run frontend/streamlit_app.py --server.port 8501
```

**Frontend**: http://localhost:8501  
**API**: http://localhost:8000

### Docker Deployment

```bash
docker-compose up --build
```

**Note**: For live deployment, you can deploy to:

- **Streamlit Cloud** (Frontend): https://share.streamlit.io
- **Render/Railway** (API): Free hosting options
- **Heroku**: Full stack deployment

---

## 3. API Endpoint

### Base URL

```
http://localhost:8000
```

### Query Endpoint

```
POST http://localhost:8000/recommend
Content-Type: application/json

{
  "query": "I need personality assessment tools",
  "top_k": 5
}
```

### Response Format (JSON)

```json
{
  "query": "I need personality assessment tools",
  "recommendations": [
    {
      "title": "Occupational Personality Questionnaire (OPQ)",
      "description": "...",
      "url": "https://www.shl.com/solutions/products/assessments/personality/opq/",
      "score": 0.85
    }
  ],
  "count": 5
}
```

### API Documentation

```
GET http://localhost:8000/docs
```

Interactive Swagger UI with all endpoints.

---

## 4. GitHub Repository

**Repository URL**: https://github.com/DeepuML/SHL-Assesment-

### Repository Contents:

- ✓ Source code (`src/`, `api/`, `frontend/`, `scraper/`, `scripts/`)
- ✓ Data files (`data/shl_catalog.json`, `data/train/`, `data/test/`)
- ✓ Evaluation results (`data/submission_optimized.csv`)
- ✓ Requirements (`requirements.txt`)
- ✓ Docker setup (`Dockerfile`, `docker-compose.yml`)
- ✓ CI/CD pipeline (`.github/workflows/ci-cd.yml`)
- ✓ README with documentation

---

## 5. Approach Document (2-page PDF)

**File to create**: `Deepu_SHL_Approach.pdf`

### Suggested Structure:

#### Page 1: Problem & Approach

1. **Problem Statement** (2-3 sentences)
2. **Architecture Overview** (diagram/flowchart)
3. **Data Collection**
   - Scraping strategy (Playwright)
   - Data preprocessing
   - Catalog size: 270+ products
4. **Embedding & Retrieval**
   - Baseline: Sentence-BERT
   - Enhanced: Google Gemini embeddings
   - FAISS indexing for fast retrieval
5. **Query Processing**
   - Text normalization
   - Semantic search
   - Ranking algorithm

#### Page 2: Evaluation & Results

1. **Evaluation Methodology**
   - Train/test split
   - Metrics: Recall@5, Precision, MRR, NDCG
   - Hyperparameter tuning
2. **Results Table**
   ```
   Model              | Recall@5 | Precision@5
   ------------------|----------|------------
   Baseline          | 18.5%    | 11.2%
   Gemini            | 19.8%    | 12.1%
   Optimized         | 21.22%   | 13.5%
   ```
3. **Key Findings**
   - Best parameters: similarity_weight=0.6, rerank_weight=0.4
   - Gemini embeddings improved by ~8%
4. **Deployment**
   - FastAPI + Streamlit
   - Docker containerization
   - CI/CD with GitHub Actions

---

## 6. Predictions CSV

**File to submit**: `Deepu_ML.csv` (your firstname_lastname.csv)

**Current file**: `data/submission_optimized.csv`

### File Format:

```csv
ID,product_1,product_2,product_3,product_4,product_5
Q001,product_url_1,product_url_2,product_url_3,product_url_4,product_url_5
Q002,product_url_1,product_url_2,product_url_3,product_url_4,product_url_5
...
```

**Action Required**: Rename the file to match your name format.

---

## Submission Checklist

- [ ] Copy `data/submission_optimized.csv` → `Deepu_ML.csv` (or your actual name)
- [ ] Create 2-page PDF document (`Deepu_SHL_Approach.pdf`)
- [ ] Verify GitHub repository is public and accessible
- [ ] Test API endpoint locally (http://localhost:8000/recommend)
- [ ] Test Frontend locally (http://localhost:8501)
- [ ] (Optional) Deploy to cloud for live URL
- [ ] Verify all files in submission portal

---

## Quick Commands

### Start Services

```bash
# API only
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Frontend only
streamlit run frontend/streamlit_app.py

# Both with Docker
docker-compose up --build
```

### Test API

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "leadership assessment", "top_k": 5}'
```

### Generate Predictions

```bash
python scripts/generate_predictions_optimized.py
```

---

## Contact Information

**GitHub**: https://github.com/DeepuML/SHL-Assesment-  
**Best Model Performance**: 21.22% Recall@5

---

**Last Updated**: December 17, 2025
