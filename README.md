# Isthmus

Isthmus is an experimental RAG project exploring whether document-local graph
retrieval can find more complete evidence than conventional top-k search. It
expands from relevant passages through bounded chunk relationships, then ranks
documents for complementary evidence and assembles a cited context.

It is a research prototype, not a production-ready service or a proven
improvement over standard RAG. Retrieval can miss evidence, and citations do
not guarantee that an answer is correct.

## Features

- PDF, DOCX, PPTX, XLSX, TXT, and Markdown ingestion (PDF OCR is not supported).
- SQLite persistence and local ChromaDB vector search.
- Bounded graph traversal, document reranking, and token-budgeted contexts.
- Streamlit UI, CLI, and local HTTP API.
- OpenAI-compatible embedding and answer endpoints; Anthropic Messages API for
  answer generation.

## Run locally

Requires Python 3.11 or newer. From PowerShell:

```powershell
git clone https://github.com/gahanlal/Contextweave.git
cd Contextweave
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ui]"
Copy-Item .env.example .env
python -m pytest
streamlit run frontend\streamlit_app.py
```

Set `ISTHMUS_EMBEDDING_MODEL` and `ISTHMUS_EMBEDDING_API_KEY` in `.env`, then
create a sector and upload documents in the UI. For generated answers, also
configure the answer provider and choose a model in the UI. Provider API calls
may incur charges; documents and questions are sent to the configured endpoint.
Never commit API keys.

For CLI PDF indexing and retrieval:

```powershell
isthmus index-pdfs --pdf-dir testfiles --sector "Research"
isthmus query "What does the report say about the project?"
```

The HTTP API runs locally with `python -m uvicorn isthmus.api:app`. It has no
authentication; do not expose it publicly.

## Evaluation

I evaluated retrieval on 77 PDFs from
[Docugami KG-RAG-datasets](https://github.com/docugami/KG-RAG-datasets), using
171 eligible SEC questions. On the token-aware MiniLM index, document-reranked
Isthmus achieved **42.9% precision, 71.2% recall, and 47.9% F1** for
document-pattern matches. Dense top-k achieved **29.4% precision, 98.2%
recall, and 39.7% F1**. Thus, Isthmus improved precision in this run but
missed more labeled source documents.

A separate seven-question passage review found complete evidence for 6/7
questions with Isthmus and 5/7 with top-k. The questions and evidence judgments
were not independently reviewed and are too few to estimate general performance.
No generated-answer quality, citation accuracy, or API cost was measured.

The results are experimental: source filename patterns are coarse labels, the
passage review is small, and single-run latency is not a production benchmark.
They do not prove that Isthmus is better than conventional RAG.

See the [full benchmark and methodology](docs/docugami-evaluation.md), the
[passage review](docs/docugami-passage-review.md), and
[ingestion/index compatibility notes](docs/ingestion.md).

## Tests

Run the unit and integration test suite with:

```powershell
python -m pytest
```

The included sample evaluation labels are illustrative, not independently
reviewed. Review and update labels when using a different document index.
