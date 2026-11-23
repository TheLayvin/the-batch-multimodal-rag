# Multimodal RAG System for *The Batch* (DeepLearning.AI)

An end-to-end **Multimodal Retrieval-Augmented Generation (RAG)** system that scrapes articles and visual media (charts, diagrams, illustrations) from [The Batch](https://www.deeplearning.ai/the-batch/), indexes both text and visual representations in a vector database, and provides grounded, source-cited answers with accompanying images through an interactive **Streamlit** UI.

---

## System Architecture

The pipeline is divided into an **Offline Indexing Pipeline** and an **Online Query Pipeline**:

```text
[Offline Pipeline]
The Batch Website --> scraper.py (BS4 + Next.js image unwrapping) --> articles.json + data/images/
                                                                                |
                                                                                v
ChromaDB (cosine) <-- vector_store.py (gemini-embedding-001) <-- processor.py (Chunking + Gemini Vision Captions)

[Online Pipeline]
User Query --> vector_store.py (Asymmetric Query Embedding + Metadata Filtering)
                     |
                     +--> Top-K Text Excerpts
                     +--> Top-K Images (filtered by similarity threshold)
                                |
                                v
                     rag_engine.py (In-memory Pillow JPEG resize + Gemini 3.5 Flash-Lite)
                                |
                                v
                     Streamlit UI (Grounded Answer + Source Links + Image Gallery)
```

---

## Technical Decisions & Reasoning

| Component | Selected Tool / Model | Reasoning |
| :--- | :--- | :--- |
| **Web Scraping** | `requests` + `BeautifulSoup4` | Unwraps Next.js `/_next/image?url=...` proxy URLs to extract full-resolution content images while filtering out site logos, SVGs, and promotional badges. |
| **Chunking Strategy** | Paragraph-based Recursive Chunking (`800` chars, `150` overlap) | Preserves sentence and paragraph boundaries instead of fixed character splits. Prepends `Article: {title}` to every chunk (*Contextual Enrichment*) so chunks retain article context. |
| **Multimodal Integration** | **VLM Visual Summarization + Dual-Stage Visual Grounding** (`gemini-3.5-flash-lite`) | News articles in *The Batch* contain dense benchmark charts and tables where raw CLIP embeddings struggle. Generating dense VLM captions enables accurate semantic retrieval, while passing the actual resized image pixels at generation time allows the LLM to read exact chart numbers. |
| **Image Preprocessing** | `Pillow` (`RGB`, `1024x1024` thumbnail, `BytesIO` JPEG) | Compresses large raw assets (including multi-MB `.gif` and `.png` files) in memory to ~80 KB JPEGs without disk clutter, reducing API latency and token overhead. |
| **Vector Database** | `ChromaDB` (`PersistentClient`, `hnsw:space: cosine`) | Lightweight local persistence with HNSW indexing and metadata filtering (`where={"type": "text"}` vs `{"type": "image"}`) so text chunks do not crowd out visual media. |
| **Embeddings** | `gemini-embedding-001` (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`) | Asymmetric task-type embeddings align short user queries with longer article excerpts. Includes automatic `429` rate-limit retry and idempotent indexing. |
| **UI Framework** | `Streamlit` | Clean two-column layout displaying the grounded answer, clickable article citations, retrieved visual media gallery, retrieval hyperparameter sliders, and expandable raw chunk observability. |

---

## System Evaluation Results

Evaluated using `src/evaluator.py` across a 6-query Golden Dataset covering benchmark comparisons, product launches, financial filings, and technical concepts (`top_k_text=4`, `top_k_images=3`, `min_image_similarity=0.45`):

| Metric | Score | Meaning |
| :--- | :---: | :--- |
| **Text Hit Rate @ 4** | **100.0%** | Target article retrieved in Top-4 text excerpts for all 6/6 queries. |
| **MRR @ 4** | **1.000** | Target article ranked #1 on every single query. |
| **Visual Hit Rate @ 3** | **66.7%** | Target article's image retrieved in Top-3 visual results for 4/6 queries. |
| **Avg Faithfulness (LLM-as-a-Judge)** | **5.0 / 5.0** | Zero hallucinations; claims strictly grounded in retrieved text and charts. |
| **Avg Answer Relevance (LLM-as-a-Judge)** | **5.0 / 5.0** | Answers directly and accurately address the user queries. |

### Error Analysis & Future Improvements
* **Why Visual Hit Rate @ 3 is 66.7%:** To respect Free Tier API quotas, VLM captioning was capped at the first `max_vlm_images=15` out of 41 images; remaining images fell back to `alt_text` only. Additionally, some newsletter issues contain decorative editorial illustrations rather than topic-specific charts, which are filtered out by `min_image_similarity`.
* **Production Scaling Roadmap:** Adding `asyncio` concurrency for VLM captioning, automatic model fallback (`gemini-3.8-flash` -> `gemini-3.5-flash-lite`), and a Cross-Encoder Reranker for large-scale corpora.

---

## Project Structure

```text
multimodal-rag-the-batch/
├── src/
│   ├── __init__.py
│   ├── config.py          # Paths, model constants, and .env loading
│   ├── scraper.py       # The Batch scraper & image downloader
│   ├── processor.py       # Semantic chunking & Gemini Vision captioning
│   ├── vector_store.py    # Batch embedding generation & ChromaDB search
│   ├── rag_engine.py      # End-to-end multimodal retrieval & generation
│   └── evaluator.py       # Retrieval (Hit Rate, MRR) & LLM-as-a-Judge evals
├── app.py                 # Streamlit interactive web UI
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Excludes .env, venv, local data, and chroma_db
└── README.md              # Documentation & technical report
```

---

## Setup & Running Instructions

### 1. Clone the repository and create a virtual environment
```bash
git clone <your-repo-url>
cd multimodal-rag-the-batch
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Copy `.env.example` to `.env` and add your Google AI Studio API key:
```bash
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_actual_key_here
```

### 4. Run the Data & Indexing Pipeline
```bash
# Step 1: Scrape articles and images from The Batch
python -m src.scraper

# Step 2: Chunk text and generate VLM image captions
python -m src.processor

# Step 3: Build the ChromaDB vector index
python -m src.vector_store
```

### 5. Launch the Web UI
```bash
streamlit run app.py
```

### 6. (Optional) Run System Evaluation
```bash
python -m src.evaluator
```

## Cloud Architecture & AWS Deployment

To keep the Git repository and Docker image stateless and lightweight (zero binary data or secrets in Git/ECR), this project separates code from data using AWS managed services:

* **Source Control (GitHub):** Pure application code (`src/`, `app.py`, `Dockerfile`), excluding `.env`, `data/`, and `chroma_db/`.
* **Data & Vector Index Storage (Amazon S3):** Pre-built `chroma_db/` and scraped `data/images/` are synced to a private S3 bucket via `src/sync_s3.py`.
* **Container Registry (Amazon ECR):** Lightweight Python 3.12 Docker image built via multi-layer caching.
* **Compute & Security (AWS EC2 + IAM Role):** Hosted on Ubuntu EC2 (`t3.micro`) using an **IAM Instance Profile** (`EC2RagS3EcrRole`) for passwordless authentication to S3 and ECR. On container startup, `src/sync_s3.py --download` pulls the index from S3 before launching Streamlit on port `8501`.

### Quick Docker & AWS Sync Commands
```bash
# Upload local data/ and chroma_db/ to S3 (requires AWS CLI & S3_BUCKET_NAME in .env)
python -m src.sync_s3 --upload

# Build and run locally with Docker
docker build -t the-batch-rag .
docker run -p 8501:8501 --env-file .env the-batch-rag
```