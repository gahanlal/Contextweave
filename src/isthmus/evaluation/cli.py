from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from isthmus.backends import open_index
from isthmus.embeddings import OpenAIEmbeddingProvider
from isthmus.evaluation.dataset import load_dataset
from isthmus.evaluation.runner import evaluate_dataset
from isthmus.evaluation.sample import build_sample_evaluation
from isthmus.pipeline import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="isthmus-evaluate")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--database")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--embedding-base-url")
    args = parser.parse_args()

    load_dotenv()
    model = args.model or os.getenv("ISTHMUS_EMBEDDING_MODEL", "")
    raw_dimensions = os.getenv("ISTHMUS_EMBEDDING_DIMENSIONS", "")
    try:
        dimensions = args.dimensions if args.dimensions is not None else (int(raw_dimensions) if raw_dimensions else None)
    except ValueError:
        parser.error("ISTHMUS_EMBEDDING_DIMENSIONS must be an integer")
    if not model.strip():
        parser.error("Set ISTHMUS_EMBEDDING_MODEL or pass --model")
    provider = OpenAIEmbeddingProvider(
        model, dimensions,
        base_url=args.embedding_base_url or os.getenv("ISTHMUS_EMBEDDING_BASE_URL") or None,
    )
    if args.demo:
        index, cases = build_sample_evaluation(provider)
        report = evaluate_dataset(index, cases)
    else:
        if not args.database or not args.dataset:
            parser.error("--database and --dataset are required unless --demo is used")
        with open_index(provider, args.database) as index:
            report = evaluate_dataset(index, load_dataset(args.dataset), PipelineConfig())
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
