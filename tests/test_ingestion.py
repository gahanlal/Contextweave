from types import SimpleNamespace

import pytest

from isthmus import ingestion
from isthmus.ingestion import _split_text, parse_pdf, parse_pdf_bytes


def test_pdf_text_split_respects_word_limit_and_preserves_content():
    chunks = _split_text(
        "one two three four five six seven eight nine ten",
        max_words=4,
    )

    assert chunks == ("one two three four", "five six seven eight", "nine ten")


def test_pdf_identity_tracks_bytes_and_requires_explicit_sector(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingestion, "PdfReader",
        lambda stream: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: stream.read().decode())]),
    )
    path = tmp_path / "Cyber_Crime_Law.pdf"
    path.write_bytes(b"First revision.")
    first, chunks, _ = parse_pdf(path, "Finance")
    renamed, renamed_chunks, _ = parse_pdf_bytes(b"First revision.", "renamed.pdf", "Finance")
    changed, changed_chunks, _ = parse_pdf_bytes(b"Second revision.", path.name, "Finance")

    assert first.id == renamed.id
    assert first.revision_id == renamed.revision_id
    assert chunks[0].id == renamed_chunks[0].id
    assert changed.id != first.id
    assert changed.revision_id != first.revision_id
    assert changed_chunks[0].id != chunks[0].id
    assert first.sector_id == chunks[0].sector_id == "Finance"
    with pytest.raises(ValueError, match="sector"):
        parse_pdf_bytes(b"First revision.", path.name, " ")
    with pytest.raises(TypeError):
        parse_pdf(path)


def test_token_aware_pdf_chunks_preserve_all_content(monkeypatch):
    class Provider:
        max_input_tokens = 10

        def count_tokens(self, text):
            return len(text) + 2

        def embed(self, text):
            return (1.0, 0.0)

    text = "one longunbrokenword two three"
    monkeypatch.setattr(
        ingestion, "PdfReader",
        lambda stream: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: text)]),
    )
    provider = Provider()
    _, chunks, relationships = parse_pdf_bytes(b"pdf", "policy.pdf", "Finance", embedding_provider=provider)
    assert len(chunks) > 1
    assert all(provider.count_tokens(chunk.content) <= provider.max_input_tokens for chunk in chunks)
    assert "".join(chunk.content.replace(" ", "") for chunk in chunks) == text.replace(" ", "")
    assert len(relationships) == 2 * (len(chunks) - 1)


def test_token_limit_too_small_is_reported():
    with pytest.raises(ValueError, match="too small"):
        _split_text("text", 300, (lambda text: len(text) + 2, 2))
