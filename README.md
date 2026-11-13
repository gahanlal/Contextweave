# Isthmus

Isthmus is a research proof of concept for testing whether adaptive, document-local graph retrieval can give a retrieval-augmented generation (RAG) system more complete evidence than conventional retrieval.

## What this project is attempting

Conventional RAG often selects the top matching chunks and sends them directly to a language model. Those chunks may contain the main rule but miss an exception, qualification, or related passage elsewhere in the document. Adding a fixed window of neighboring chunks can recover some context, but may also include irrelevant material and still miss evidence beyond that window.

Isthmus investigates a different approach:

1. Search the whole corpus for matching chunks, then use those matches to identify relevant sectors and candidate documents.
2. Expand independently within each candidate document's chunk graph. The traversal can pass through a low-similarity bridge without including it, potentially finding relevant context beyond it.
3. Only after every candidate has been expanded, compare complete document evidence packages. Prefer documents that add useful, complementary evidence rather than selecting solely by individual chunk similarity.
4. Assemble a bounded context with source locations and use it to generate an answer with citations.

For example, a policy's main rule and its exception might be separated by an unrelated paragraph. The experiment asks whether bounded graph expansion can find both without blindly including the intermediate paragraph. Document-level reranking then asks whether a second document contributes a distinct qualification or merely repeats the first.

The goal is to compare evidence completeness, document selection, context size, and latency against top-k and fixed-window retrieval—not to assume that more graph traversal is always better. The project is experimental: approximate vector search, traversal limits, and reranking can miss evidence, and citations are not proof that an answer is correct.

## Current implementation

The initial implementation focuses on the retrieval mechanism:

1. A global vector search returns matching chunks across all indexed sectors.
2. Deterministic scoring routes to sectors and discovers candidate documents.
3. Each candidate document is expanded independently with bounded bidirectional traversal.
4. Traversal records visited, included, and bridged chunks separately; `QUALIFIES` edges are enabled for exception-oriented queries.
5. After every candidate has an evidence package, document-level reranking selects complementary evidence.
6. Context assembly prioritizes marginal requirement coverage, preserves source locations and enforces a global token budget.

The retrieval core uses SQLite for durable documents and graph relationships and an in-memory ChromaDB collection for vector search. Choose an embedding model offered by an OpenAI-compatible embeddings API. Answer generation supports OpenAI-compatible chat APIs and Anthropic's Messages API. Models are not selected by default; API calls may incur charges.

### Why SQLite and ChromaDB?

They were chosen for convenience during the proof of concept: SQLite needs no separate database server, and an in-memory ChromaDB collection makes it straightforward to try vector search without setting up PostgreSQL and pgvector. This is a setup choice, not a claim that either is the best production option. ChromaDB's collection is rebuilt from the stored SQLite vectors.

### Bring your own database

SQLite remains the default. To integrate another database, such as PostgreSQL, implement a subclass of `isthmus.index.BackendIndex` and a factory `create_index(embedding_provider, database)` returning it. Make the module importable and set `ISTHMUS_INDEX_FACTORY=your_module:create_index` in `.env`; the UI, CLI, API, and evaluation runner all use this factory. The factory can obtain connection credentials from its own environment variables rather than putting secrets in the UI or command line.

The backend must provide the document, chunk, sector, relationship and version state used by retrieval; implement sector and document lifecycle methods, context-manager cleanup and vector search, and increment `version` after index changes. It must preserve embedding-provider identity, chunk IDs, provenance, and document-local relationships. See the built-in `SQLiteIndex` and `tests/test_backends.py` for the expected interface. The extension point executes trusted local Python code; do not load untrusted factories.

PostgreSQL and pgvector themselves are **not included**: a developer must implement and test the adapter, schema, connection handling, and vector search for their chosen database. The small factory mechanism does not automatically convert an existing SQLite index.

## Run locally

Isthmus is not deployed as a hosted service. Clone this repository and run it on your own machine; the default SQLite and ChromaDB setup does not require a separate database server. You will need access to the model APIs you choose.

Use Python 3.11 or newer. On Windows, replace `python` below with `py -3.14` if `python` points to an older installation.

```powershell
git clone https://github.com/gahanlal/isthmus-rag.git
cd isthmus-rag
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest
python -m streamlit run frontend\streamlit_app.py
```

In `.env`, set `ISTHMUS_EMBEDDING_MODEL` and `ISTHMUS_EMBEDDING_API_KEY` to your chosen model ID and provider key. For an OpenAI-compatible provider other than the default endpoint, set `ISTHMUS_EMBEDDING_BASE_URL`; set `ISTHMUS_EMBEDDING_DIMENSIONS` only if your model and existing index require it. The optional `OPENAI_API_KEY` is also accepted. Use separate databases for different model, dimension, or endpoint combinations. Do not commit keys.

For answers, select the API protocol and enter a model ID in the UI. Set `ISTHMUS_ANSWER_API_KEY` (or the provider's `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) and optionally `ISTHMUS_ANSWER_BASE_URL`. Custom endpoints must implement the selected API protocol; arbitrary APIs are not automatically compatible. Anthropic's Messages API is for answers, not embeddings. Documents, questions, and evidence are sent to the configured providers.

### Try the included sample

1. In Streamlit, enter the embedding model and endpoint and create a sector.
2. Upload both files from `sample_data/` (the two `.txt` files) and index them in that sector.
3. Use the query tab to inspect retrieval and, optionally, generate an answer.
4. In the evaluation tab, upload `sample_data/evaluation.jsonl` and run the five baselines.

The sample is fictional and included for redistribution. Its four labels are illustrative, not independently reviewed or sufficient to establish benchmark quality. Keep the filenames and default chunking so their IDs match the labels. To try the older synthetic command-line demo, run `isthmus-demo`; it uses your configured embedding provider and incurs any applicable API charges.

The example PDFs and the local database are not distributed with this repository. Add PDFs you have permission to use to `testfiles/`, then index them into `isthmus.db` and run a retrieval query:

```bash
python -m isthmus.cli index-pdfs --sector "Your sector"
python -m isthmus.cli query "What cybercrime laws and exceptions apply?"
```

Existing installations can keep using their previous database file: pass its path with `--database`, enter it in the Streamlit sidebar, or set `ISTHMUS_DATABASE` for the HTTP API. Select the same embedding model and dimensions used when it was indexed. A provider mismatch is rejected rather than mixing incompatible vectors; create a separate database to change models. Existing databases and settings are not automatically moved or overwritten.

The index command requires a manually chosen sector and reports PDFs without extractable text; OCR is not implemented. Different sectors require separate batches. Documents now use content-based IDs: changed content is indexed as a separate revision, and re-indexing existing filename-based records does not replace them. See [ingestion compatibility](docs/ingestion.md) before re-indexing or using older evaluation labels. The query command prints retrieved evidence and provenance. Add `--answer --answer-protocol <openai|anthropic> --answer-model <model-id>` to generate an answer. Neither the demo nor a single query is a quality evaluation, and no claim is made that Isthmus outperforms conventional retrieval.

## Requirement coverage and optional query planner

The default query planner is deterministic and splits obvious comparisons and separately phrased questions; it will still miss many implicit requirements. For an optional model-based planner, set `ISTHMUS_QUERY_PLANNER_MODEL`, and optionally `ISTHMUS_QUERY_PLANNER_PROTOCOL` (`openai` or `anthropic`) and `ISTHMUS_QUERY_PLANNER_BASE_URL`. It uses the answer-provider API key; it sends the query to that provider and may incur extra charges. Invalid model output or provider request failures are logged and fall back to the deterministic planner. This feature is opt-in and does not guarantee a correct plan.

For each query requirement, the retrieval pipeline compares its embedding with each evidence chunk and combines similarity with lexical support; exception coverage also requires explicit exception language and overlap with query-topic terms in the same chunk. Scores are aggregated per requirement across included chunks, rather than treating one shared word as full coverage. The thresholds are experimental heuristics, not calibrated evidence judgments: semantic similarity does not prove a claim, and implicit exceptions, paraphrases, or exceptions whose topic is only in an adjacent chunk can still be missed. Requirement embeddings may add API calls and latency. Final coverage reports `sufficient`, `partial`, or `insufficient` according to these heuristics, and partial or insufficient coverage can trigger one bounded fallback. The context budget uses the stored chunk token counts, which are word-count estimates rather than model-specific tokenization.

## Streamlit interface

Install the optional UI dependency and launch the app:

```bash
python -m pip install -e ".[dev,ui]"
streamlit run frontend/streamlit_app.py
```

The UI supports sector creation, PDF, DOCX, PPTX, XLSX, TXT, and Markdown upload/re-index/removal, retrieval, optional answers, evidence inspection, traversal visualization, and JSONL evaluation datasets. PDF extraction depends on extractable text; OCR is not provided. Office document page layout is not inferred, so DOCX citations identify document blocks rather than pages. Spreadsheet rows retain cell coordinates and slide citations retain slide numbers.

Generated citations use identifiers assigned to retrieved evidence and are checked against the assembled context. A second call to the selected answer model assesses whether each factual claim is cited and supported; unsupported claims and malformed assessments are rejected. This model-based check can make mistakes and is not independent verification; users should check cited evidence. Answer generation now makes two model calls, adding latency and potentially cost.

## HTTP API

Run `python -m uvicorn isthmus.api:app` and visit its `/docs` endpoint for document upload, sector creation, deletion, and retrieval. Set `ISTHMUS_DATABASE` to select the SQLite file and configure the embedding environment variables above. For generated answers, include `answer_protocol`, `answer_model`, and optionally `answer_base_url` in the query request. The API is intended for local use and has no authentication; do not expose it publicly.

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
python -m isthmus.evaluation.cli --database isthmus.db --dataset evaluation.jsonl
```

The runner compares top-k, fixed-window, adaptive graph, sector-routed graph, and document-reranked retrieval. It reports evidence/exception recall, requirement coverage, document selection precision/recall/F1, chunk/token counts, and retrieval latency. Answer correctness and citation correctness are not yet measured. The three-query demo is only a smoke test; credible comparisons still require a larger manually labeled corpus.

### Real-document retrieval experiment

We tested the local prototype against PDFs and question/source annotations from [Docugami's KG-RAG-datasets](https://github.com/docugami/KG-RAG-datasets). After token-aware re-indexing and a selection change, the document-reranked pipeline on 171 eligible SEC questions had higher source-pattern precision than dense top-k (42.9% vs. 29.4%) but lower recall (71.2% vs. 98.2%). These are document-pattern matches, **not** answer accuracy or proof of evidence coverage. See the [experiment report](docs/docugami-evaluation.md) for historical results, comparisons and limitations.

A separate [assistant-authored passage review](docs/docugami-passage-review.md) checked returned text for seven questions across three sectors. On the token-aware index, top-k supplied complete evidence for 5/7 and the full pipeline for 6/7. These small, non-independently-reviewed results do not establish general performance or generated-answer quality.

`evaluation/pdf-draft.jsonl` contains nine draft cases derived from a local PDF collection that is not included in the repository. These labels were prepared by the assistant, not independently human-reviewed; they are illustrative, not a representative benchmark. They only work with the same documents, filenames and chunk boundaries; otherwise, create labels for your own index. Review the labels and expand the corpus before using the results as evidence of retrieval quality. With the matching collection indexed, run:

```bash
python -m isthmus.evaluation.cli --database isthmus.db --dataset evaluation/pdf-draft.jsonl
```

This sends the questions to the configured embedding provider and may incur charges.

## Performance characteristics

- ChromaDB uses a local in-memory HNSW index for approximate nearest-neighbor search. The index is rebuilt from SQLite vectors when needed; retrieval results may differ from exact search.
- Vectors are normalized once when indexed. A query is embedded once, and graph expansion reuses that vector; each traversed chunk's similarity is computed at most once per query.
- The document graph is cached and rebuilt only when the index changes. Traversal remains bounded by `TraversalConfig`.
- Context assembly is a single pass over selected evidence rather than collecting and sorting it again.
- SQLite stores vectors as packed 64-bit floats (`8 * d` bytes) instead of decimal JSON. Loading uses ordered batched reads, and relationship batches commit in one transaction.
- The SQLite vector cache and ChromaDB index both occupy memory. Index build cost and query latency should be measured on the target corpus; no GPU acceleration is currently enabled.

## Main extension points

- `EmbeddingProvider`: use `OpenAIEmbeddingProvider` with a compatible embeddings endpoint, or implement the protocol for another API.
- `SQLiteIndex`: persist documents, chunks, relationships, and embeddings; build an in-memory ChromaDB collection for retrieval.
- `TraversalConfig`: tune node, depth, bridge, direction, inclusion, and token limits.
- `Isthmus`: orchestration entry point; graph expansion completes for all candidates before document reranking starts.

Structure extraction is best-effort: PDF heading hierarchy and table layout are not reconstructed, and OCR is not included.
