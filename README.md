# ContextWeave

ContextWeave is a research POC for testing adaptive, document-local graph retrieval. Its core hypothesis is that bounded expansion through a document's chunk graph can recover useful context across low-similarity bridges, while reranking complete document evidence packages can favor complementary evidence over redundant high-scoring chunks.

## Current implementation

The initial implementation focuses on the retrieval mechanism:

1. A global vector search returns matching chunks across all indexed sectors.
2. Deterministic scoring routes to sectors and discovers candidate documents.
3. Each candidate document is expanded independently with bounded bidirectional traversal.
4. Traversal records visited, included, and bridged chunks separately; `QUALIFIES` edges are enabled for exception-oriented queries.
5. After every candidate has an evidence package, document-level reranking selects complementary evidence.
6. Context assembly preserves source locations and enforces a global token budget.

The repository is deliberately starting with this core before database, parser, model-provider, and UI integrations. The bundled `HashingEmbeddingProvider` is a deterministic lexical baseline for local runs, **not** a semantic embedding model. It implements the same provider interface that a configured embedding service can replace.

## Run locally

```bash
python -m pip install -e ".[dev]"
pytest
contextweave-demo
```

The demo prints an inspectable JSON retrieval trace using a small example policy. It is not a quality evaluation and makes no claim that ContextWeave outperforms conventional retrieval.

## Main extension points

- `EmbeddingProvider`: replace the local lexical baseline with a semantic embedding provider.
- `InMemoryIndex`: replace or complement the local store with PostgreSQL/pgvector persistence.
- `TraversalConfig`: tune node, depth, bridge, direction, inclusion, and token limits.
- `ContextWeave`: orchestration entry point; graph expansion completes for all candidates before document reranking starts.

The next integration phase should add structure-aware ingestion and PostgreSQL/pgvector persistence while retaining the tested retrieval interfaces.
