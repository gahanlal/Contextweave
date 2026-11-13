from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sqlite3
from typing import Annotated
from zipfile import BadZipFile

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from openai import OpenAIError
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError

from isthmus.document_parsing import parse_document_bytes
from isthmus.embeddings import HashingEmbeddingProvider, OpenAIEmbeddingProvider
from isthmus.generation import OpenAIAnswerProvider, generate_answer
from isthmus.index import SQLiteIndex
from isthmus.pipeline import Isthmus

load_dotenv()
app = FastAPI(title="Isthmus")


class SectorRequest(BaseModel):
    name: str = Field(min_length=1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    generate_answer: bool = False


def _index() -> SQLiteIndex:
    provider_name = os.getenv("ISTHMUS_EMBEDDINGS", "openai")
    if provider_name == "openai":
        provider = OpenAIEmbeddingProvider()
    elif provider_name == "hashing":
        provider = HashingEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
    return SQLiteIndex(provider, Path(os.getenv("ISTHMUS_DATABASE", "isthmus.db")))


@app.get("/documents")
def list_documents() -> dict:
    with _index() as index:
        return {"sectors": sorted(index.sectors), "documents": [asdict(doc) for doc in index.documents.values()]}


@app.post("/sectors", status_code=201)
def create_sector(request: SectorRequest) -> dict:
    try:
        with _index() as index:
            index.add_sector(request.name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"name": request.name.strip()}


@app.post("/documents", status_code=201)
def upload_document(
    file: Annotated[UploadFile, File()],
    sector: Annotated[str, Form()],
) -> dict:
    try:
        if not file.filename:
            raise ValueError("A document filename is required.")
        with _index() as index:
            if sector not in index.sectors:
                raise ValueError("Create the sector before uploading a document.")
            document, chunks, relationships = parse_document_bytes(
                file.file.read(), file.filename, sector
            )
            index.add_document(document, chunks, relationships)
            return {"document": asdict(document), "chunks": len(chunks)}
    except (ValueError, UnicodeDecodeError, BadZipFile, PdfReadError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OpenAIError as error:
        raise HTTPException(status_code=502, detail=f"Embedding provider error: {type(error).__name__}") from error


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    with _index() as index:
        try:
            index.remove_document(document_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
    return {"deleted": document_id}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be blank.")
    try:
        with _index() as index:
            if not index.chunks:
                raise HTTPException(status_code=400, detail="Index at least one document.")
            result = Isthmus(index).query(request.question)
            answer = (
                generate_answer(result, index.documents, OpenAIAnswerProvider())
                if request.generate_answer else None
            )
            trace = asdict(result)
            for hit in trace["global_hits"]:
                hit["chunk"].pop("embedding")
            for chunk in trace["context"]:
                chunk.pop("embedding")
            return {"retrieval": jsonable_encoder(trace), "answer": asdict(answer) if answer else None}
    except OpenAIError as error:
        raise HTTPException(status_code=502, detail=f"Model provider error: {type(error).__name__}") from error
    except (ValueError, sqlite3.Error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
