# Kanoon-RAG — Income Tax Assistant

A Retrieval-Augmented Generation (RAG) application that answers Indian income tax questions, grounded strictly in the text of the Income-tax Act, 2025. Built to explore production RAG engineering: retrieval tuning, hallucination control, and multilingual prompt design.

**Live demo:** https://kanoon-rag.onrender.com

> Note: hosted on Render's free tier, so the first request after a period of inactivity may take 30–60 seconds (cold start).

---

## What it does

Ask a question about Indian income tax — in English, Hindi, or Hinglish — and get an answer sourced only from the actual legal text, with the relevant section cited. If the answer isn't in the source material, the assistant says so explicitly instead of guessing.

Currently covers:
- Section 123 — Life insurance, PF & investment deductions (formerly 80C)
- Section 126 — Health insurance premium deduction (formerly 80D)
- Section 129 — Education loan interest deduction (formerly 80E)
- Section 130 — Home loan interest deduction
- Section 133 — Donations (formerly 80G)
- Section 153 — Interest on savings deposits (formerly 80TTA/80TTB)
- Section 115BAC — New Tax Regime slabs and Section 87A rebate
- Old Tax Regime slabs and exemptions

All source text was collected directly from [incometaxindia.gov.in](https://incometaxindia.gov.in), the official Income Tax Department portal, under the Income-tax Act, 2025 (effective 1 April 2026).

## Tech stack

| Layer | Tools |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), Hugging Face |
| Vector store | ChromaDB |
| LLM | Groq API (`openai/gpt-oss-20b`) |
| Backend | FastAPI, Uvicorn |
| Chunking | LangChain text splitters |
| Frontend | Vanilla HTML/CSS/JS |
| Containerization | Docker (CPU-only build) |
| Deployment | Render |

## Architecture

```
User question
    │
    ▼
Embedding (sentence-transformers)
    │
    ▼
Vector search (ChromaDB, top-k retrieval)
    │
    ▼
Retrieved chunks + question → prompt
    │
    ▼
LLM (Groq) generates answer, grounded in retrieved text only
    │
    ▼
Response (language-matched, cited, plain text)
```

## Engineering decisions worth noting

A few problems came up during development that shaped the final design:

- **Chunk size vs. retrieval coverage.** Small chunks (500 chars) split long clause lists across many pieces, so broad questions ("what investments qualify under 123?") returned incomplete answers. Increasing chunk size to 1500 with 200-char overlap fixed this, but a later increase in document count degraded retrieval again — solved by scaling `n_results` with the number of indexed documents rather than keeping it fixed.

- **Hallucination control.** The prompt explicitly restricts the model to facts present in the retrieved context, requires it to decline with a fixed fallback line when the context doesn't cover the question, and forbids mixing a real answer with the fallback disclaimer in the same response.

- **Section-number retrieval gap.** Queries that referenced only a section number (e.g., "Section 115BAC kya hai") sometimes failed even when the data existed, because the number appeared only in document metadata, not in the body text the embedding model indexes. Fixed by repeating the section number explicitly in the body of each source file.

- **Citation consistency.** Source files include both the new (2025 Act) and old (1961 Act) section numbers for continuity, which caused the model to sometimes cite the outdated number. A citation rule was added requiring the current section number as the primary reference.

- **Incremental indexing.** Rather than rebuilding the vector database from scratch on every data update, `build_index.py` hashes each source file and only re-embeds files that are new or changed, skipping unchanged ones.

- **CPU-only deployment.** The default PyTorch install pulls in CUDA/GPU dependencies that are unnecessary for an app with no GPU access, which significantly slowed down Docker builds. Pinning the CPU-only PyTorch wheel cut build time and image size considerably.

## Running locally

```bash
git clone <repo-url>
cd kanoon-rag
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# add your Groq API key to a .env file:
# GROQ_API_KEY=your_key_here

python build_index.py          # builds the vector database from /data
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000`.

## Running with Docker

```bash
docker build -t kanoon-rag .
docker run -p 8000:8000 --env-file .env kanoon-rag
```

## Project structure

```
├── data/                   # Source legal text files (one per section/topic)
├── static/
│   └── index.html          # Frontend chat interface
├── main.py                 # FastAPI backend + RAG pipeline
├── build_index.py          # Incremental indexing script
├── ask.py                  # CLI tool for quick local testing
├── requirements.txt
├── Dockerfile
└── README.md
```

## Limitations

- Coverage is limited to the sections listed above — it is not a full replacement for a tax professional, and the UI says so.
- Tax slabs and rates can change with each Finance Act; the data here reflects FY 2026-27 as sourced at the time of writing and should be reverified for later years.
- Hosted on a free tier, so response times and uptime are best-effort, not production SLA.

## License

For educational and portfolio purposes.
