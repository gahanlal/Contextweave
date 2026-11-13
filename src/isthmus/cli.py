from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from isthmus.backends import open_index
from isthmus.embeddings import OpenAIEmbeddingProvider
from isthmus.generation import answer_provider, generate_answer
from isthmus.ingestion import NoExtractableTextError, parse_pdf
from isthmus.pipeline import Isthmus


def _sector_argument(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("sector must not be empty")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(prog="isthmus")
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser(
        "index-pdfs",
        help="Extract PDFs, create online embeddings, and store them in SQLite.",
    )
    index_parser.add_argument("--pdf-dir", type=Path, default=Path("testfiles"))
    index_parser.add_argument("--sector", required=True, type=_sector_argument,
                              help="Explicit sector assigned to every PDF in this batch.")
    index_parser.add_argument("--database", type=Path, default=Path("isthmus.db"))
    index_parser.add_argument("--model")
    index_parser.add_argument("--dimensions", type=int)
    index_parser.add_argument("--embedding-base-url")

    query_parser = commands.add_parser(
        "query",
        help="Run retrieval against a previously indexed SQLite database.",
    )
    query_parser.add_argument("question", nargs="+")
    query_parser.add_argument("--database", type=Path, default=Path("isthmus.db"))
    query_parser.add_argument("--model")
    query_parser.add_argument("--dimensions", type=int)
    query_parser.add_argument("--embedding-base-url")
    query_parser.add_argument("--answer", action="store_true", help="Generate a cited answer.")
    query_parser.add_argument("--answer-protocol", choices=("openai", "anthropic"), default="openai")
    query_parser.add_argument("--answer-model")
    query_parser.add_argument("--answer-base-url")

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
    if args.command == "query" and args.answer and not (args.answer_model or os.getenv("ISTHMUS_ANSWER_MODEL")):
        parser.error("Set ISTHMUS_ANSWER_MODEL or pass --answer-model when using --answer")
    provider = OpenAIEmbeddingProvider(
        model=model,
        dimensions=dimensions,
        base_url=args.embedding_base_url or os.getenv("ISTHMUS_EMBEDDING_BASE_URL") or None,
    )

    if args.command == "index-pdfs":
        _index_pdfs(args.pdf_dir, args.database, provider, args.sector)
    else:
        _query(
            " ".join(args.question), args.database, provider, args.answer,
            args.answer_protocol, args.answer_model or os.getenv("ISTHMUS_ANSWER_MODEL", ""),
            args.answer_base_url or os.getenv("ISTHMUS_ANSWER_BASE_URL") or None,
        )


def _index_pdfs(
    pdf_directory: Path,
    database_path: Path,
    provider: OpenAIEmbeddingProvider,
    sector_id: str,
) -> None:
    if not pdf_directory.is_dir():
        raise NotADirectoryError(f"PDF directory does not exist: {pdf_directory}")
    pdf_paths = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")

    indexed_documents = 0
    indexed_chunks = 0
    skipped_files: list[str] = []
    with open_index(provider, database_path) as index:
        for path in pdf_paths:
            try:
                document, chunks, relationships = parse_pdf(path, sector_id, embedding_provider=index.embedding_provider)
            except NoExtractableTextError as error:
                print(f"Skipped {path.name}: {error}", file=sys.stderr)
                skipped_files.append(path.name)
                continue
            index.add_document(document, chunks, relationships)
            indexed_documents += 1
            indexed_chunks += len(chunks)
            print(f"Indexed {path.name}: {len(chunks)} chunks ({document.sector_id})")

    print(
        json.dumps(
            {
                "database": str(database_path),
                "documents_indexed": indexed_documents,
                "chunks_indexed": indexed_chunks,
                "files_skipped": skipped_files,
            },
            indent=2,
        )
    )


def _query(
    question: str,
    database_path: Path,
    provider: OpenAIEmbeddingProvider,
    answer: bool = False,
    answer_protocol: str = "openai",
    answer_model: str = "",
    answer_base_url: str | None = None,
) -> None:
    with open_index(provider, database_path) as index:
        if not index.chunks:
            raise RuntimeError(f"No chunks are indexed in {database_path}")
        result = Isthmus(index).query(question)
        generated = (
            generate_answer(result, index.documents, answer_provider(answer_protocol, answer_model, answer_base_url))
            if answer else None
        )
        print(
            json.dumps(
                {
                    "query": result.query_plan.original_query,
                    "candidate_documents": result.candidate_document_ids,
                    "selected_documents": result.selected_document_ids,
                    "evidence_coverage": result.evidence_coverage,
                    "answer": generated.text if generated else None,
                    "citations": [
                        {
                            "id": citation.id,
                            "document_id": citation.document_id,
                            "title": citation.document_title,
                            "location": citation.location,
                            "chunk_id": citation.chunk_id,
                        }
                        for citation in (generated.citations if generated else ())
                    ],
                    "context": [
                        {
                            "document_id": chunk.document_id,
                            "location": chunk.source_location,
                            "content": chunk.content,
                        }
                        for chunk in result.context
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
