"""Retrieve contexts for the assistant-authored Docugami passage review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docugami_benchmark import (
    DeterministicPlanner,
    LocalMiniLM,
    TokenAwareMiniLM,
    match_context_words,
)
from isthmus.evaluation.baselines import run_baselines
from isthmus.index import SQLiteIndex
from isthmus.pipeline import Isthmus, PipelineConfig


CASES = (
    ("apple", "According to the 2023 Q3 Apple filing, what were net sales and net income for the three months ended July 1, 2023?"),
    ("amazon", "According to the 2023 Q3 Amazon filing, what were net sales and operating income for the three months ended September 30, 2023?"),
    ("comparison", "Compare the three-month net sales reported in the filings labeled 2023 Q3 for Apple and Amazon."),
    ("aviation-cause", "What did the NTSB determine was the probable cause of the Piper PA-22-150 accident near Tallahassee?"),
    ("aviation-weather", "What visibility and ceiling were reported for the Reims F172M accident near Chevroux, France?"),
    ("trial-size", "How many participants were planned for the single-center oral chronic graft-versus-host disease study using mesenchymal stem cell dressings?"),
    ("trial-vr", "For the spinal-injury virtual-reality rehabilitation feasibility study, what was the intended intervention schedule and target recruitment?"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--legacy-chunking", action="store_true", help="Use the original word-only index's provider.")
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error("The indexed database must already exist.")
    provider = LocalMiniLM() if args.legacy_chunking else TokenAwareMiniLM()
    config = PipelineConfig()
    cases = []
    with SQLiteIndex(provider, args.database) as index:
        pipeline = Isthmus(index, config, query_planner=DeterministicPlanner())
        for case_id, query in CASES:
            methods = []
            baselines = {result.name: result for result in run_baselines(index, query, config, pipeline)}
            top_k, isthmus = match_context_words(
                baselines["A: top-k"].chunks,
                baselines["E: document reranking"].chunks,
                config.global_token_budget,
            )
            for name, result, chunks in (
                ("A: top-k (size-matched)", baselines["A: top-k"], top_k),
                ("E: document reranking (size-matched)", baselines["E: document reranking"], isthmus),
            ):
                methods.append({
                    "method": name,
                    "latency_ms": result.latency_ms,
                    "context_words": sum(chunk.token_count for chunk in chunks),
                    "chunks": [
                        {
                            "document": index.documents[chunk.document_id].title,
                            "page": chunk.source_location,
                            "chunk_id": chunk.id,
                            "text": chunk.content,
                        }
                        for chunk in chunks
                    ],
                })
            cases.append({"id": case_id, "question": query, "methods": methods})
            print(f"Retrieved {case_id}", flush=True)
    args.report.write_text(
        json.dumps(
            {
                "embedding_provider": provider.provider_id,
                "document_selection_policy": config.document_selection_policy,
                "context_matching": "Both methods use the smaller available per-question whitespace-word count.",
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
