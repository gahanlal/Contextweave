from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from isthmus.embeddings import HashingEmbeddingProvider, OpenAIEmbeddingProvider
from isthmus.evaluation.dataset import load_dataset
from isthmus.evaluation.runner import evaluate_dataset
from isthmus.evaluation.sample import build_sample_evaluation
from isthmus.index import SQLiteIndex
from isthmus.pipeline import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="isthmus-evaluate")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--database")
    parser.add_argument("--dataset")
    parser.add_argument("--provider", choices=("openai", "hashing"), default="openai")
    args = parser.parse_args()

    if args.demo:
        index, cases = build_sample_evaluation()
        report = evaluate_dataset(index, cases)
    else:
        if not args.database or not args.dataset:
            parser.error("--database and --dataset are required unless --demo is used")
        load_dotenv()
        provider = (
            OpenAIEmbeddingProvider()
            if args.provider == "openai"
            else HashingEmbeddingProvider()
        )
        with SQLiteIndex(provider, args.database) as index:
            report = evaluate_dataset(index, load_dataset(args.dataset), PipelineConfig())
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
