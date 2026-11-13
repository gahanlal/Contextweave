# Isthmus

Isthmus is a research POC for testing adaptive, document-local graph retrieval. Its core hypothesis is that bounded expansion through a document's chunk graph can recover useful context across low-similarity bridges, while reranking complete document evidence packages can favor complementary evidence over redundant high-scoring chunks.

## Current implementation

The initial implementation focuses on the retrieval mechanism:

1. A global vector search returns matching chunks across all indexed sectors.
2. Deterministic scoring routes to sectors and discovers candidate documents.
3. Each candidate document is expanded independently with bounded bidirectional traversal.
4. Traversal records visited, included, and bridged chunks separately; `QUALIFIES` edges are enabled for exception-oriented queries.
5. After every candidate has an evidence package, document-level reranking selects complementary evidence.
6. Context assembly preserves source locations and enforces a global token budget.

The retrieval core uses SQLite for durable documents and graph relationships and an in-memory ChromaDB collection for vector search. The default embedding model is `text-embedding-3-small`; OpenAI API usage may incur charges. `HashingEmbeddingProvider` remains available for deterministic offline tests and demos, but is not a semantic embedding model.

## Run locally

```bash
python -m pip install -e ".[dev]"
pytest
isthmus-demo
```

Set `OPENAI_API_KEY` in the environment or a local, ignored `.env` file to use online embeddings. Indexing sends extracted PDF text to OpenAI, and API usage may incur charges. The key is read locally and must not be committed.

The example PDFs and the local database are not distributed with this repository. Add PDFs you have permission to use to `testfiles/`, then index them into `isthmus.db` and run a retrieval query:

```bash
python -m isthmus.cli index-pdfs
python -m isthmus.cli query "What cybercrime laws and exceptions apply?"
```

Existing installations can keep using their previous database file: pass its path with `--database`, enter it in the Streamlit sidebar, or set `ISTHMUS_DATABASE` for the HTTP API. The new default does not automatically move or overwrite existing data. The API configuration variables are now `ISTHMUS_DATABASE` and `ISTHMUS_EMBEDDINGS`.

The index command reports PDFs without extractable text; OCR is not implemented. The query command prints retrieved evidence and provenance. Add `--answer` to generate a cited answer with OpenAI `gpt-4o-mini`; this incurs API charges. The separate demo still prints an inspectable trace for a small synthetic policy. Neither is a quality evaluation, and no claim is made that Isthmus outperforms conventional retrieval.

## Streamlit interface

Install the optional UI dependency and launch the app:

```bash
python -m pip install -e ".[dev,ui]"
streamlit run frontend/streamlit_app.py
```

The UI supports sector creation, PDF, DOCX, PPTX, XLSX, TXT, and Markdown upload/re-index/removal, retrieval, optional `gpt-4o-mini` answers, evidence inspection, traversal visualization, and JSONL evaluation datasets. Select the local hashing provider for offline retrieval; it is a lexical baseline, not a semantic model. Answer generation still requires OpenAI API access, even when using hashing. PDF extraction depends on extractable text; OCR is not provided. Office document page layout is not inferred, so DOCX citations identify document blocks rather than pages. Spreadsheet rows retain cell coordinates and slide citations retain slide numbers.

Generated citations use identifiers assigned to retrieved evidence and are checked against the assembled context. This verifies that source identifiers exist, not that every claim is supported; users should check cited evidence.

## HTTP API

Run `uvicorn isthmus.api:app` and visit its `/docs` endpoint for document upload, sector creation, deletion, and retrieval. Set `ISTHMUS_DATABASE` to select the SQLite file and `ISTHMUS_EMBEDDINGS` to `openai` (default) or `hashing`. The API is intended for local use and has no authentication; do not expose it publicly.

## Retrieval evaluation

Run the built-in three-query synthetic smoke dataset:

```bash
python -m isthmus.evaluation.cli --demo
```

For corpus evaluation, provide JSONL with one manually labeled case per line:

```json
{"id":"case-1","query":"What exceptions apply?","evidence_chunk_ids":["chunk-17"],"exception_chunk_ids":["chunk-17"],"supporting_document_ids":["doc-2"],"requirement_chunk_ids":{"exception":["chunk-17"]}}
```

Then run:

```bash
python -m isthmus.evaluation.cli --database isthmus.db --dataset evaluation.jsonl --provider hashing
```

The runner compares top-k, fixed-window, adaptive graph, sector-routed graph, and document-reranked retrieval. It reports evidence/exception recall, requirement coverage, document selection precision/recall/F1, chunk/token counts, and retrieval latency. Answer correctness and citation correctness are not yet measured. The three-query demo is only a smoke test; credible comparisons still require a larger manually labeled corpus.

`evaluation/pdf-draft.jsonl` contains nine draft cases derived from a local PDF collection that is not included in the repository. These labels were prepared by the assistant, not independently human-reviewed; they are illustrative, not a representative benchmark. They only work with the same documents, filenames and chunk boundaries; otherwise, create labels for your own index. Review the labels and expand the corpus before using the results as evidence of retrieval quality. With the matching collection indexed, run:

```bash
python -m isthmus.evaluation.cli --database isthmus.db --dataset evaluation/pdf-draft.jsonl --provider openai
```

This sends the questions to OpenAI for embeddings and may incur charges.

## Performance characteristics

- ChromaDB uses a local in-memory HNSW index for approximate nearest-neighbor search. The index is rebuilt from SQLite vectors when needed; retrieval results may differ from exact search.
- Vectors are normalized once when indexed. A query is embedded once, and graph expansion reuses that vector; each traversed chunk's similarity is computed at most once per query.
- The document graph is cached and rebuilt only when the index changes. Traversal remains bounded by `TraversalConfig`.
- Context assembly is a single pass over selected evidence rather than collecting and sorting it again.
- SQLite stores vectors as packed 64-bit floats (`8 * d` bytes) instead of decimal JSON. Loading uses ordered batched reads, and relationship batches commit in one transaction.
- The SQLite vector cache and ChromaDB index both occupy memory. Index build cost and query latency should be measured on the target corpus; no GPU acceleration is currently enabled.

## Main extension points

- `EmbeddingProvider`: use `OpenAIEmbeddingProvider` for online vectors or `HashingEmbeddingProvider` for offline tests.
- `SQLiteIndex`: persist documents, chunks, relationships, and embeddings; build an in-memory ChromaDB collection for retrieval.
- `TraversalConfig`: tune node, depth, bridge, direction, inclusion, and token limits.
- `Isthmus`: orchestration entry point; graph expansion completes for all candidates before document reranking starts.

Structure extraction is best-effort: PDF heading hierarchy and table layout are not reconstructed, and OCR is not included.
