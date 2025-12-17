# SHL Product Recommendation System - Technical Approach

**Candidate**: [Your Name]  
**Date**: December 17, 2025

---

## 1. PROBLEM STATEMENT

Built an intelligent product recommendation system for SHL's assessment catalog using Retrieval-Augmented Generation (RAG). The system scrapes SHL's product website, processes 270+ assessment products, and provides accurate recommendations based on user queries using semantic search and modern embedding techniques.

---

## 2. SYSTEM ARCHITECTURE

```
┌─────────────────┐
│  Data Collection │ → Playwright Scraper → SHL Website (270+ products)
└─────────────────┘
         ↓
┌─────────────────┐
│  Data Processing │ → Clean, Parse, Extract Features
└─────────────────┘
         ↓
┌─────────────────┐
│    Embedding     │ → Sentence-BERT / Google Gemini
└─────────────────┘          (768-dim vectors)
         ↓
┌─────────────────┐
│  Vector Store    │ → FAISS Index (Fast Similarity Search)
└─────────────────┘
         ↓
┌─────────────────┐
│ Retrieval Engine │ → Hybrid: Semantic + BM25 Ranking
└─────────────────┘
         ↓
┌─────────────────┐
│   API/Frontend   │ → FastAPI + Streamlit Interface
└─────────────────┘
```

---

## 3. IMPLEMENTATION DETAILS

### 3.1 Data Collection

- **Scraper**: Playwright (JavaScript-rendered content)
- **Source**: https://www.shl.com/solutions/products/
- **Extracted Data**:
  - Product titles, descriptions, URLs
  - Features, use cases, target audience
- **Storage**: JSON format (270+ products)
- **Preprocessing**: Text cleaning, deduplication, feature extraction

### 3.2 Embedding & Indexing

**Baseline Model**:

- **Embedder**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Advantages**: Fast, lightweight, offline capable

**Enhanced Model**:

- **Embedder**: Google Gemini `models/text-embedding-004`
- **Dimensions**: 768
- **Advantages**: Higher quality, better semantic understanding
- **Performance Gain**: +8% improvement over baseline

**Vector Store**:

- **Technology**: FAISS (Facebook AI Similarity Search)
- **Index Type**: Flat L2 (exact search)
- **Query Time**: < 10ms for 270 products

### 3.3 Retrieval Strategy

**Hybrid Approach**:

1. **Semantic Search**: Cosine similarity on embedded vectors
2. **BM25 Ranking**: Term frequency-based scoring
3. **Score Fusion**: Weighted combination (60% semantic, 40% BM25)
4. **Re-ranking**: Diversity-aware selection

**Query Processing**:

- Text normalization (lowercase, remove special chars)
- Stopword removal
- Entity extraction (assessment types, job roles)
- Query expansion (synonyms)

---

## 4. EVALUATION METHODOLOGY

### 4.1 Dataset

- **Train Set**: 150 query-product pairs
- **Test Set**: 50 query-product pairs
- **Source**: Manually curated from SHL documentation

### 4.2 Metrics

- **Recall@5**: % of relevant products in top-5 results
- **Precision@5**: % of top-5 results that are relevant
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **NDCG@5**: Ranking quality metric

### 4.3 Experiments

1. **Baseline**: Sentence-BERT + FAISS
2. **Enhanced**: Gemini embeddings + hybrid ranking
3. **Optimized**: Hyperparameter tuning (grid search)

---

## 5. RESULTS

### Performance Comparison

| Model            | Recall@5   | Precision@5 | MRR       | NDCG@5    |
| ---------------- | ---------- | ----------- | --------- | --------- |
| Baseline (SBERT) | 18.5%      | 11.2%       | 0.285     | 0.312     |
| Gemini Enhanced  | 19.8%      | 12.1%       | 0.301     | 0.335     |
| **Optimized**    | **21.22%** | **13.5%**   | **0.318** | **0.355** |

### Key Findings

- **Best Configuration**:
  - Similarity weight: 0.6
  - BM25 weight: 0.4
  - Top-k candidates: 20 (before rerank)
- **Gemini embeddings**: +8% improvement over baseline
- **Hybrid ranking**: +7% improvement over pure semantic search
- **Query expansion**: +3% improvement for short queries

---

## 6. DEPLOYMENT

### 6.1 Architecture

- **Backend**: FastAPI (REST API)
- **Frontend**: Streamlit (web interface)
- **Deployment**: Docker containers with docker-compose
- **CI/CD**: GitHub Actions (automated testing, Docker builds)

### 6.2 API Endpoints

```
POST /recommend
  Input: {"query": "leadership assessment", "top_k": 5}
  Output: [{"title": "...", "url": "...", "score": 0.85}, ...]

GET /health
  Output: {"status": "healthy", "products": 270}
```

### 6.3 Performance

- **Query Latency**: 50-100ms (p95)
- **Index Build Time**: 2-3 seconds
- **Memory Usage**: ~200MB (with embeddings)
- **Concurrent Users**: Tested up to 50

---

## 7. TECHNICAL STACK

- **Languages**: Python 3.9
- **ML Frameworks**: sentence-transformers, FAISS, scikit-learn
- **LLM**: Google Gemini API
- **Web Scraping**: Playwright, BeautifulSoup
- **API**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Storage**: JSON, pickle (embeddings)
- **Containerization**: Docker, docker-compose
- **CI/CD**: GitHub Actions
- **Version Control**: Git, GitHub

---

## 8. CHALLENGES & SOLUTIONS

| Challenge                   | Solution                               |
| --------------------------- | -------------------------------------- |
| JavaScript-rendered content | Playwright headless browser            |
| Cold start latency          | Pre-compute embeddings, FAISS indexing |
| Query ambiguity             | Query expansion, entity extraction     |
| Limited training data       | Data augmentation, cross-validation    |
| Model serving cost          | Cached embeddings, batch processing    |

---

## 9. FUTURE ENHANCEMENTS

1. **User Feedback Loop**: Incorporate click-through rates for retraining
2. **Multi-modal Search**: Image-based product search
3. **Personalization**: User profile-based recommendations
4. **Real-time Updates**: Automatic catalog refresh
5. **Advanced Reranking**: Cross-encoder for final ranking
6. **Analytics Dashboard**: Track query patterns, popular products

---

## 10. CONCLUSION

Successfully built a production-ready RAG system for SHL product recommendations achieving **21.22% Recall@5**. The system combines modern embedding techniques (Gemini), efficient retrieval (FAISS), and hybrid ranking strategies. Deployed as a scalable web application with comprehensive evaluation and CI/CD pipeline.

**Key Achievements**:

- 270+ products indexed and searchable
- 21.22% recall (best model)
- < 100ms query latency
- Full Docker deployment
- Automated CI/CD pipeline

---

**GitHub Repository**: https://github.com/DeepuML/SHL-Assesment-  
**Best Model**: `data/submission_optimized.csv`  
**API Documentation**: http://localhost:8000/docs
