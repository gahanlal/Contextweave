from contextweave.ingestion import _split_text, sector_for_pdf


def test_pdf_text_split_respects_word_limit_and_preserves_content():
    chunks = _split_text(
        "one two three four five six seven eight nine ten",
        max_words=4,
    )

    assert chunks == ("one two three four", "five six seven eight", "nine ten")


def test_sector_is_inferred_from_pdf_filename():
    assert sector_for_pdf("Cyber_Crime_Law_and_Practice.pdf") == "Cybersecurity & Law"
    assert sector_for_pdf("Sustainable Management.pdf") == "Sustainability"
