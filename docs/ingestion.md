# Ingestion identity and compatibility

PDF CLI indexing requires a sector for the whole batch, for example
`isthmus index-pdfs --pdf-dir testfiles --sector "Finance"`.
Run separate batches for different sectors. Filenames are not used to infer sectors.
The UI and API also require an explicit sector.

Newly parsed documents use `sha256-<full SHA-256 of source bytes>` as their
document ID and the full digest as `revision_id`. All supported formats use the
same rule. Identical bytes have the same ID even under different filenames;
indexing them again replaces that indexed revision, including its title, sector,
chunks and relationships. Different bytes, even with the same filename or the
same extracted text, create separate revisions that coexist. Revisions are
content-addressed records; there is no automatic grouping or ordering of
revisions of the same logical document. Chunk IDs include the document ID, and
also depend on parsing and chunking settings.

Existing indexed databases are not migrated or deleted. Their old filename-based
IDs and revision values remain usable. Re-indexing a previously indexed file
creates a new content-addressed record alongside the old one; remove the old
record manually only if desired. Old document/chunk evaluation labels and
citations still refer to the old records, not the new ones. Update labels after
re-indexing and verify their evidence and chunk boundaries; the distributed
`sample_data/evaluation.jsonl` labels target the newly parsed sample documents.
`evaluation/pdf-draft.jsonl` retains its legacy IDs for existing PDF indexes;
it must be relabeled against the particular PDFs and index before evaluating
newly indexed revisions.

Chunking normally uses a word limit, which is not the same as a model token
limit. An embedding provider may additionally expose both `count_tokens(text)`
and a positive integer `max_input_tokens`. Ingestion then splits text to fit
that provider's limit, and checked embedding calls reject over-limit inputs.
Providers must count all tokens relevant to their model, including special
tokens. The built-in OpenAI-compatible provider does not currently advertise
this capability, so its limits are not automatically enforced. An existing
index must be re-indexed to change chunk boundaries; check and update its
chunk-level labels and citations. Use a distinct provider identity or database
when embedding compatibility changes.
