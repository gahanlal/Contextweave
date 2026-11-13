from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from isthmus.embeddings import OpenAIEmbeddingProvider
from isthmus.generation import OpenAIAnswerProvider, generate_answer
from isthmus.index import SQLiteIndex
from isthmus.ingestion import NoExtractableTextError, parse_pdf
from isthmus.pipeline import Isthmus


def main() -> None:
    parser = argparse.ArgumentParser(prog="isthmus")
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser(
        "index-pdfs",
        help="Extract PDFs, create online embeddings, and store them in SQLite.",
    )
    index_parser.add_argument("--pdf-dir", type=Path, default=Path("testfiles"))
    index_parser.add_argument("--database", type=Path, default=Path("isthmus.db"))
    index_parser.add_argument("--model", default="text-embedding-3-small")

    query_parser = commands.add_parser(
        "query",
        help="Run retrieval against a previously indexed SQLite database.",
    )
    query_parser.add_argument("question", nargs="+")
    query_parser.add_argument("--database", type=Path, default=Path("isthmus.db"))
    query_parser.add_argument("--model", default="text-embedding-3-small")
    query_parser.add_argument("--answer", action="store_true", help="Generate a cited answer using OpenAI.")

    args = parser.parse_args()
    load_dotenv()
    provider = OpenAIEmbeddingProvider(model=args.model)

    if args.command == "index-pdfs":
        _index_pdfs(args.pdf_dir, args.database, provider)
    else:
        _query(" ".join(args.question), args.database, provider, args.answer)


def _index_pdfs(
    pdf_directory: Path,
    database_path: Path,
    provider: OpenAIEmbeddingProvider,
) -> None:
    if not pdf_directory.is_dir():
        raise NotADirectoryError(f"PDF directory does not exist: {pdf_directory}")
    pdf_paths = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_directory}")

    indexed_documents = 0
    indexed_chunks = 0
    skipped_files: list[str] = []
    with SQLiteIndex(provider, database_path) as index:
        for path in pdf_paths:
            try:
                document, chunks, relationships = parse_pdf(path)
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
) -> None:
    with SQLiteIndex(provider, database_path) as index:
        if not index.chunks:
            raise RuntimeError(f"No chunks are indexed in {database_path}")
        result = Isthmus(index).query(question)
        generated = (
            generate_answer(result, index.documents, OpenAIAnswerProvider())
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
