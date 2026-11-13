"""Reproduce the document-level Docugami retrieval comparison locally."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import statistics
import time
from pathlib import Path

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from pypdf.errors import PyPdfError
from tokenizers import Tokenizer

from isthmus.evaluation.baselines import run_baselines
from isthmus.index import SQLiteIndex
from isthmus.ingestion import NoExtractableTextError, parse_pdf
from isthmus.pipeline import Isthmus, PipelineConfig
from isthmus.query import understand_query


SECTORS = (
    "sec-10-q",
    "ntsb-aviation-incident-accident-reports",
    "nih-clinical-trial-protocols",
    "us-fed-agency-reports",
)


class LocalMiniLM:
    provider_id = "chromadb-onnx-all-MiniLM-L6-v2:384"

    def __init__(self) -> None:
        self.model = ONNXMiniLM_L6_V2()

    def embed_many(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        return tuple(tuple(float(value) for value in vector) for vector in self.model(texts))

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_many([text])[0]


class TokenAwareMiniLM(LocalMiniLM):
    provider_id = "chromadb-onnx-all-MiniLM-L6-v2:384:token-aware-v1"
    max_input_tokens = 256

    def __init__(self) -> None:
        super().__init__()
        tokenizer_path = self.model.DOWNLOAD_PATH / self.model.EXTRACTED_FOLDER_NAME / "tokenizer.json"
        if not tokenizer_path.is_file():
            self.model(["Initialize tokenizer"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)


class DeterministicPlanner:
    def plan(self, query: str):
        return understand_query(query)


def budget_chunks(chunks, budget):
    selected = []
    used = 0
    for chunk in chunks:
        if used + chunk.token_count <= budget:
            selected.append(chunk)
            used += chunk.token_count
    return tuple(selected)


def document_metrics(expected, actual):
    expected, actual = set(expected), set(actual)
    if not expected:
        raise ValueError("Source pattern matched no documents.")
    overlap = len(expected & actual)
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "any_hit": float(overlap > 0),
        "all_sources": float(overlap == len(expected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--comparators", action="store_true", help="Include hybrid, cross-encoder and page-merge baselines.")
    parser.add_argument("--legacy-chunking", action="store_true", help="Use the original word-only benchmark chunking.")
    args = parser.parse_args()
    root = args.data_dir.resolve()
    files = {
        sector: sorted((root / sector / "data" / "v1" / "docs").glob("*.pdf"))
        for sector in SECTORS
    }
    if any(not paths for paths in files.values()):
        raise ValueError("Each sector must contain at least one PDF.")
    csv_path = root / "sec-10-q" / "data" / "v1" / "qna_data.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        cases = list(csv.DictReader(stream))
    expected_paths = {path.name: path for path in files["sec-10-q"]}
    if len(expected_paths) != len(files["sec-10-q"]):
        raise ValueError("SEC filenames must be unique.")

    provider = LocalMiniLM() if args.legacy_chunking else TokenAwareMiniLM()
    config = PipelineConfig()
    indexed = []
    failed = []
    results = []
    excluded = []
    started = time.perf_counter()
    with SQLiteIndex(provider, args.database) as index:
        corpus_ids = {
            "sha256-" + hashlib.sha256(path.read_bytes()).hexdigest()
            for paths in files.values() for path in paths
        }
        if set(index.documents) - corpus_ids:
            raise ValueError("Database contains documents outside this corpus; use a separate database.")
        for sector, paths in files.items():
            for path in paths:
                document_id = "sha256-" + hashlib.sha256(path.read_bytes()).hexdigest()
                try:
                    if document_id not in index.documents:
                        document, chunks, relationships = parse_pdf(path, sector, embedding_provider=provider)
                        index.add_document(document, chunks, relationships)
                    document = index.documents[document_id]
                    if document.sector_id != sector or document.title != path.stem:
                        raise ValueError(f"Existing document metadata does not match {path.name}.")
                    indexed.append({
                        "sector": sector,
                        "path": str(path.relative_to(root)).replace("\\", "/"),
                        "document_id": document_id,
                        "chunks": sum(chunk.document_id == document_id for chunk in index.chunks.values()),
                    })
                    print(f"Indexed {len(indexed)}: {sector}/{path.name}", flush=True)
                except (NoExtractableTextError, PyPdfError) as error:
                    failed.append({
                        "sector": sector,
                        "path": str(path.relative_to(root)).replace("\\", "/"),
                        "error": f"{type(error).__name__}: {error}",
                    })
                    print(f"Skipped {sector}/{path.name}: {type(error).__name__}", flush=True)

        filename_to_id = {
            Path(item["path"]).name: item["document_id"]
            for item in indexed if item["sector"] == "sec-10-q"
        }
        pipeline = Isthmus(index, config, query_planner=DeterministicPlanner())
        if args.comparators:
            from docugami_comparators import CrossEncoder, HybridSearch, PageMerger, run_comparators

            hybrid = HybridSearch(index.chunks.values())
            merger = PageMerger(index.chunks.values())
            cross_encoder = CrossEncoder()
        for number, case in enumerate(cases, start=1):
            pattern = case["Source Docs"].strip()
            matching = sorted(name for name in expected_paths if fnmatch.fnmatchcase(name, pattern))
            if not matching:
                raise ValueError(f"Case {number} source pattern matches no SEC PDFs: {pattern}")
            missing = [name for name in matching if name not in filename_to_id]
            if missing:
                excluded.append({"case": number, "pattern": pattern, "missing": missing})
                continue
            expected = {filename_to_id[name] for name in matching}
            baselines = run_baselines(index, case["Question"], config, pipeline)
            if args.comparators:
                baselines += run_comparators(
                    index, case["Question"], hybrid, cross_encoder, merger, config.global_top_k
                )
            for baseline in baselines:
                chunks = budget_chunks(baseline.chunks, config.global_token_budget)
                actual = list(dict.fromkeys(chunk.document_id for chunk in chunks))
                results.append({
                    "case": number,
                    "question": case["Question"],
                    "question_type": case["Question Type"],
                    "source_pattern": pattern,
                    "expected_files": matching,
                    "baseline": baseline.name,
                    "retrieved_files": [index.documents[doc].title + ".pdf" for doc in actual],
                    "chunks": len(chunks),
                    "estimated_words": sum(chunk.token_count for chunk in chunks),
                    "latency_ms": baseline.latency_ms,
                    **document_metrics(expected, actual),
                })
            print(f"Evaluated case {number}/{len(cases)}", flush=True)

        summary = {}
        for baseline in sorted({row["baseline"] for row in results}):
            baseline_rows = [row for row in results if row["baseline"] == baseline]
            summary[baseline] = {
                "cases": len(baseline_rows),
                **{
                    key: statistics.mean(row[key] for row in baseline_rows)
                    for key in ("precision", "recall", "f1", "any_hit", "all_sources", "chunks", "estimated_words")
                },
                "median_latency_ms": statistics.median(row["latency_ms"] for row in baseline_rows),
            }
        report = {
            "dataset": "https://github.com/docugami/KG-RAG-datasets",
            "dataset_commit": "547f132d3c2bcc2b9976ca1d8413b99d8ba45aa1",
            "embedding_model": provider.provider_id,
            "embedding_max_tokens": 256,
            "total_seconds": time.perf_counter() - started,
            "total_chunks": len(index.chunks),
            "indexed_files": indexed,
            "failed_files": failed,
            "source_cases": len(cases),
            "excluded_cases": excluded,
            "summary": summary,
            "per_case": results,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "failed_files": failed, "excluded_cases": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
