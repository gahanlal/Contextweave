from __future__ import annotations

from collections import Counter
from io import StringIO
import json
from pathlib import Path
import sqlite3
from zipfile import BadZipFile

from dotenv import load_dotenv
from openai import OpenAIError
from pypdf.errors import PdfReadError
import streamlit as st

from isthmus.embeddings import HashingEmbeddingProvider, OpenAIEmbeddingProvider
from isthmus.document_parsing import parse_document_bytes
from isthmus.evaluation.dataset import load_dataset_stream
from isthmus.evaluation.runner import evaluate_dataset
from isthmus.generation import OpenAIAnswerProvider, generate_answer
from isthmus.index import SQLiteIndex
from isthmus.ingestion import NoExtractableTextError
from isthmus.pipeline import Isthmus


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Isthmus", layout="wide")
    st.title("Isthmus")
    st.caption("Adaptive graph retrieval with inspectable evidence and cited answers.")

    database_path = st.sidebar.text_input("SQLite database", "isthmus.db")
    provider_name = st.sidebar.radio(
        "Embedding provider",
        ("Offline hashing (demo)", "OpenAI text-embedding-3-small"),
    )
    try:
        provider = (
            HashingEmbeddingProvider()
            if provider_name.startswith("Offline")
            else OpenAIEmbeddingProvider()
        )
    except RuntimeError as error:
        st.error(str(error))
        st.stop()
        return

    try:
        with SQLiteIndex(provider, database_path) as index:
            index_identity = (str(Path(database_path).resolve()), index.provider_id)
            if st.session_state.get("index_identity") != index_identity:
                st.session_state.pop("retrieval_result", None)
                st.session_state.pop("generated_answer", None)
                st.session_state.pop("evaluation_report", None)
                st.session_state["index_identity"] = index_identity
            st.caption(
                f"Embedding provider: `{index.provider_id}` · "
                f"{len(index.documents)} documents · {len(index.chunks)} chunks"
            )
            documents_tab, retrieval_tab, evaluation_tab = st.tabs(
                ("Documents & sectors", "Query & retrieval trace", "Evaluation")
            )
            with documents_tab:
                _documents_page(index)
            with retrieval_tab:
                _retrieval_page(index)
            with evaluation_tab:
                _evaluation_page(index)
    except (sqlite3.Error, ValueError) as error:
        st.error(str(error))


def _documents_page(index: SQLiteIndex) -> None:
    st.subheader("Sectors")
    with st.form("create_sector", clear_on_submit=True):
        sector_name = st.text_input("New sector name")
        create_sector = st.form_submit_button("Create sector")
    if create_sector:
        try:
            index.add_sector(sector_name)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state.pop("retrieval_result", None)
            st.rerun()

    if index.sectors:
        st.write(", ".join(sorted(index.sectors)))
        empty_sectors = sorted(
            sector
            for sector in index.sectors
            if not any(doc.sector_id == sector for doc in index.documents.values())
        )
        if empty_sectors:
            sector_to_remove = st.selectbox("Remove empty sector", empty_sectors)
            if st.button("Delete sector"):
                index.remove_sector(sector_to_remove)
                st.rerun()
    else:
        st.info("Create a sector before uploading documents.")

    st.subheader("Upload and index documents")
    uploads = st.file_uploader(
        "Documents",
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md"],
        accept_multiple_files=True,
        key="pdf_uploads",
    )
    sector_options = sorted(index.sectors)
    if sector_options:
        selected_sector = st.selectbox("Assign uploaded documents to", sector_options)
        if st.button("Index selected documents", disabled=not uploads):
            indexed = 0
            for upload in uploads or ():
                try:
                    document, chunks, relationships = parse_document_bytes(
                        upload.getvalue(),
                        upload.name,
                        selected_sector,
                    )
                    index.add_document(document, chunks, relationships)
                except (NoExtractableTextError, UnicodeDecodeError, ValueError, BadZipFile, PdfReadError, OpenAIError, sqlite3.Error) as error:
                    st.error(f"Could not index {upload.name}: {error}")
                    break
                indexed += 1
            if indexed:
                st.success(f"Indexed {indexed} document(s). Re-uploading a file replaces its previous index.")
                st.session_state.pop("retrieval_result", None)
                st.session_state.pop("generated_answer", None)

    st.subheader("Indexed documents")
    if index.documents:
        chunk_counts = Counter(chunk.document_id for chunk in index.chunks.values())
        st.dataframe(
            [
                {
                    "Title": document.title,
                    "Document ID": document.id,
                    "Sector": document.sector_id,
                    "Revision": document.revision_id,
                    "Chunks": chunk_counts[document.id],
                    "Status": "Indexed",
                }
                for document in sorted(index.documents.values(), key=lambda item: item.title.casefold())
            ],
            use_container_width=True,
            hide_index=True,
        )
        selected_document = st.selectbox(
            "Remove document",
            sorted(index.documents),
            format_func=lambda document_id: index.documents[document_id].title,
        )
        if st.button("Remove selected document"):
            index.remove_document(selected_document)
            st.session_state.pop("retrieval_result", None)
            st.session_state.pop("generated_answer", None)
            st.rerun()
    else:
        st.info("No documents indexed yet.")


def _retrieval_page(index: SQLiteIndex) -> None:
    query = st.text_input("Ask a question")
    generate = st.checkbox("Generate an answer using GPT-4o mini", value=False)
    if st.button("Retrieve evidence", disabled=not query.strip()):
        if not index.chunks:
            st.warning("Index at least one document before querying.")
        else:
            try:
                st.session_state.pop("retrieval_result", None)
                st.session_state.pop("generated_answer", None)
                result = Isthmus(index).query(query)
                st.session_state["retrieval_result"] = result
                st.session_state.pop("generated_answer", None)
                if generate:
                    st.session_state["generated_answer"] = generate_answer(
                        result, index.documents, OpenAIAnswerProvider()
                    )
            except (OpenAIError, ValueError) as error:
                st.error(f"Query failed: {error}")

    result = st.session_state.get("retrieval_result")
    if result is None:
        return
    answer = st.session_state.get("generated_answer")
    if answer:
        st.subheader("Answer")
        st.write(answer.text)
        for citation in answer.citations:
            st.write(f"[{citation.id}] {citation.document_title} · {citation.location} · {citation.chunk_id}")
    st.subheader("Retrieved evidence")
    st.write("Evidence coverage:", result.evidence_coverage)
    for chunk in result.context:
        title = index.documents[chunk.document_id].title
        with st.expander(f"{title} · {chunk.source_location}"):
            st.write(chunk.content)

    with st.expander("Retrieval trace", expanded=True):
        st.markdown("**1. Global matching chunks**")
        st.dataframe(
            [
                {
                    "Rank": hit.rank,
                    "Chunk": hit.chunk.id,
                    "Document": hit.chunk.document_id,
                    "Sector": hit.chunk.sector_id,
                    "Similarity": round(hit.similarity, 4),
                    "Location": hit.chunk.source_location,
                }
                for hit in result.global_hits
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**2. Sector scores and selected sectors**")
        st.json(
            {
                sector: {
                    "score": round(score, 4),
                    "selected": sector in result.selected_sector_ids,
                }
                for sector, score in result.sector_scores.items()
            }
        )
        st.markdown("**3. Candidate documents and graph expansion**")
        st.write("Fallback used:", result.fallback_used)
        st.write("Expansion order:", result.expansion_order)
        for package in result.evidence_packages:
            title = index.documents[package.document_id].title
            with st.expander(f"{title} · {package.document_id}"):
                st.write("Seeds:", package.seed_chunk_ids)
                st.write("Visited:", package.visited_chunk_ids)
                st.write("Included:", package.included_chunk_ids)
                st.write("Supported requirements:", sorted(package.supported_requirements))
                st.write(package.explanation)
                st.graphviz_chart(_graph_dot(index, package))
        st.markdown("**4. Document evidence scores**")
        st.dataframe(
            [
                {
                    "Document": assessment.document_id,
                    "Score": round(assessment.score, 4),
                    "Relevance": round(assessment.relevance_score, 4),
                    "Coverage": round(assessment.coverage_score, 4),
                    "Selected": assessment.selected,
                    "Reason": assessment.explanation,
                }
                for assessment in result.document_assessments
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.write("Final selected documents:", result.selected_document_ids)


def _graph_dot(index: SQLiteIndex, package) -> str:
    node_names = {
        chunk_id: f"node_{position}"
        for position, chunk_id in enumerate(package.visited_chunk_ids)
    }
    included = set(package.included_chunk_ids)
    seeds = set(package.seed_chunk_ids)
    lines = ["digraph retrieval {", "rankdir=LR;"]
    for chunk_id, node_name in node_names.items():
        chunk = index.chunks[chunk_id]
        color = "palegreen" if chunk_id in seeds else (
            "lightblue" if chunk_id in included else "khaki"
        )
        label = f"{chunk.sequence_number}: {chunk.source_location}"
        lines.append(
            f"{node_name} [label={json.dumps(label)}, style=filled, fillcolor={json.dumps(color)}];"
        )
    edges = set()
    for path in package.paths:
        for source, target, relation in zip(
            path.chunk_ids,
            path.chunk_ids[1:],
            path.relationship_types,
        ):
            edges.add((source, target, relation.value))
    for source, target, relation in sorted(edges):
        if source in node_names and target in node_names:
            lines.append(
                f"{node_names[source]} -> {node_names[target]} "
                f"[label={json.dumps(relation)}];"
            )
    lines.append("}")
    return "\n".join(lines)


def _evaluation_page(index: SQLiteIndex) -> None:
    st.subheader("Compare retrieval baselines")
    st.caption(
        "Upload JSONL with manually labeled supporting chunk IDs. "
        "Answer correctness requires separate human judgments."
    )
    dataset_file = st.file_uploader(
        "Labeled evaluation dataset",
        type=["jsonl", "txt"],
        key="evaluation_dataset",
    )
    if st.button("Run evaluation", disabled=dataset_file is None):
        try:
            text = dataset_file.getvalue().decode("utf-8-sig")
            cases = load_dataset_stream(StringIO(text))
            report = evaluate_dataset(index, cases)
        except (UnicodeDecodeError, ValueError, OpenAIError) as error:
            st.error(f"Evaluation failed: {error}")
        else:
            st.session_state["evaluation_report"] = report.to_dict()

    report = st.session_state.get("evaluation_report")
    if report:
        st.write(f"Queries evaluated: {report['case_count']}")
        st.dataframe(report["summary"], use_container_width=True)
        with st.expander("Per-query results"):
            st.json(report["per_case"])


if __name__ == "__main__":
    main()
