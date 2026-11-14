# Docugami retrieval experiment

## Scope and setup

This is an experimental local retrieval comparison, not a claim that Isthmus
outperforms conventional RAG. The documents and question/source annotations come
from [Docugami KG-RAG-datasets](https://github.com/docugami/KG-RAG-datasets),
commit `547f132d3c2bcc2b9976ca1d8413b99d8ba45aa1`; see the upstream MIT license.
The source PDFs and their labels are not included in this repository.

The index contained **77 PDFs and 4,964 chunks** across four sectors. All
sectors participated in global retrieval, but the quantitative questions were
only from the SEC collection: 171 of 195 questions were eligible. Twenty-four
were excluded because their source patterns included one of two SEC PDFs that
pypdf could not parse within its array safety limit. A scanned NIH PDF was
removed from the test corpus as requested.

The embeddings were generated locally with Chroma's pretrained
`all-MiniLM-L6-v2` ONNX model (384 dimensions), not a hashing or synthetic
embedding. The experiment used a separate SQLite database and Chroma vector
index, the deterministic query planner, default pipeline settings, and a
4,500-word estimated context budget for each baseline. The optional
cross-encoder was `cross-encoder/ms-marco-MiniLM-L6-v2`, quantized ONNX,
revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, on CPU. The model
downloads are external; inference for this experiment was local.

The upstream `Source Docs` column specifies filename patterns. We matched those
patterns against the SEC filenames and scored the resulting document sets.
These patterns can be broader than the documents actually needed to answer a
specific question; they do not label supporting passages. Accordingly, the
numbers below measure **pattern-matched document retrieval**, not factual
correctness, evidence entailment, or citation accuracy.

## Historical results: word-only chunking

Macro averages across the 171 eligible questions; precision, recall and F1 are
percentages. Latency is median local retrieval time in milliseconds.

| Retrieval method | Precision | Recall | F1 | Median ms |
|---|---:|---:|---:|---:|
| Dense top-k | 32.5 | 96.2 | 43.0 | 38.2 |
| Dense + fixed window | 37.3 | 90.2 | 47.3 | 42.0 |
| Adaptive graph | 41.5 | 82.0 | 49.0 | 76.0 |
| Sector routing + graph | 41.5 | 82.0 | 49.0 | 74.5 |
| Isthmus document reranking | 44.2 | 63.3 | 46.3 | 101.7 |
| Hybrid BM25 + dense, reciprocal-rank fusion | 24.7 | 97.5 | 35.3 | 42.4 |
| Hybrid + cross-encoder, top 10 passages | 33.9 | 91.7 | 44.1 | 1,950.6 |
| Dense + page auto-merging | 33.4 | 96.2 | 43.9 | 36.3 |

The page-merging comparator groups PDF chunks by page and expands a page when
at least two hits cover at least half its chunks; it is a simple page-level
approximation, not a general hierarchical auto-merging implementation. The
cross-encoder reranks up to 30 hybrid candidates but retains only 10 passages;
it used an average 2,311 estimated words versus 4,445 for dense top-k. Thus
their context sizes and output counts differ despite sharing the same maximum
budget. Timing is a warm local run on one machine; it is not a controlled
production latency or throughput benchmark. Graph setup accounting and sequential
execution differ between implementations.

## Findings

- Document reranking returned fewer pattern-unmatched documents than top-k, but
  also missed more expected documents. It used about 3,917 estimated words per
  query on average versus 4,445 for top-k. The adaptive-graph-only variants had
  the highest overall F1 in this particular comparison, not the full pipeline.
- For the 50 multi-document questions, document reranking reached 60.5% recall
  and 69.3% F1 versus 97.0% and 79.5% for top-k. For the 70 single-document,
  single-chunk questions, its document F1 was 37.1% versus 30.0% for top-k.
  These are coarse document-label results; they do not establish which chunks
  actually answer the questions.
- Hybrid retrieval improved overall pattern recall over top-k but retrieved
  more pattern-unmatched documents. The cross-encoder recovered precision over
  hybrid alone while costing substantially more local CPU time. Neither model
  was calibrated to determine factual entailment.
- The original PDF chunker used up to 300 **words**, while the embedding model
  truncates at 256 model tokens. An offline tokenizer check found 3,444 of
  4,964 chunks (69.4%) exceeded that model limit. Retrieval may therefore miss
  text near the end of those chunks.
- Two Microsoft PDFs exceeded pypdf's safety limit, and pypdf reported missing
  optional `fontTools` for some federal-report fonts. Extraction quality and
  table layout were not independently audited.
- No answers were generated. Answer correctness, claim-level citation support,
  end-to-end answer latency, and model/API cost were **not measured**. Local
  inference incurred no model API charges; electricity and hardware costs
  were not measured. These results cannot establish whether an answer is
  correct or whether a citation supports its claim.

## Historical token-aware comparison (unmatched context sizes)

After the historical run, the evaluation adapter started splitting chunks
using the actual MiniLM tokenizer and 256-token limit. The new index has the
same 77 PDFs and 171 eligible SEC questions, but
9,456 chunks. Directly counting every indexed chunk with the untruncated
tokenizer found a maximum of 256 tokens and **zero over the limit**.

The following are macro document-pattern scores from an earlier local rerun
(November 14, 2026), over the same 171 eligible questions. These natural-output
contexts were not size-matched, so the table is retained for historical
reference and must not be used as a controlled head-to-head comparison. The
first run used the token-aware index and included all local comparators. A
separate A/B run changed only the full pipeline's sector selector to the
revised GPT-4o-mini prompt; it kept the query planner deterministic. That run
completed 171 selector requests with zero fallbacks.

| Retrieval method | Precision | Recall | F1 | Mean context words | Median latency ms |
|---|---:|---:|---:|---:|---:|
| Dense top-k | 29.3 | 98.2 | 39.6 | 4,149 | 48.1 |
| Dense + fixed window | 35.6 | 93.7 | 46.0 | 4,492 | 57.3 |
| Adaptive graph | 41.8 | 81.3 | **49.2** | 4,486 | 94.6 |
| Heuristic sector routing + graph | 41.8 | 81.3 | **49.2** | 4,486 | 92.0 |
| Isthmus document reranking (heuristic sectors) | 42.9 | 71.2 | 47.9 | 4,469 | 126.5 |
| Isthmus document reranking (GPT-4o-mini sectors) | 42.9 | 71.2 | 47.9 | 4,467 | 1,491.3 |
| Hybrid BM25 + vector | 19.7 | 98.8 | 30.0 | 4,202 | 54.4 |
| Hybrid + cross-encoder | 32.9 | 90.5 | 42.7 | 1,460 | 2,022.3 |
| Page auto-merge | 30.3 | 98.2 | 40.7 | 4,246 | 45.0 |

The LLM sector selector changed document-pattern F1 by only **+0.01 percentage
points** versus heuristic selection, while adding **1,364.8 ms** to median
retrieval latency. This run provides no evidence that the LLM selector improves
these retrieval KPIs. Adaptive graph had the best F1; dense top-k had the best
recall. The full pipeline's lower recall reflects its more selective document
set.

Precision, recall, and F1 here score retrieved document filenames against
upstream `Source Docs` filename patterns. These labels are coarse and do not
measure passage entailment or answer correctness. Context length is the mean
number of whitespace-delimited words selected under the 4,500-word budget.
Latencies are single-machine, single-run measurements; the GPT-4o-mini number
includes the live sector-selection API call. They are not controlled production
latency estimates.

This earlier retrieval-only run did not generate answers. Its word counts and
metrics are historical, not the matched-context results below.

## Matched-context document-pattern comparison

The matched evaluation uses the same token-aware index, 171 eligible questions,
and per-question source-document labels. For each policy, the top-k and
document-reranking outputs were truncated to the exact whitespace-word count of
the smaller context for that question. Thus methods have equal total words
within each pair, although the fixed-target pair and marginal-coverage pair
have different mean budgets. The word counts are whitespace-delimited words,
not the answer model's tokenizer tokens.

| Selection policy | Method | Precision | Recall | F1 | Mean context words | Median latency ms |
|---|---|---:|---:|---:|---:|---:|
| Fixed target | Top-k, size-matched | 29.23 | 98.25 | 39.54 | 4,163.56 | 45.28 |
| Fixed target | Document reranking, size-matched | 43.66 | 69.74 | 47.93 | 4,163.56 | 116.47 |
| Marginal coverage | Top-k, size-matched | 37.25 | 94.01 | 47.45 | 1,811.41 | 49.75 |
| Marginal coverage | Document reranking, size-matched | 50.00 | 32.75 | 36.02 | 1,811.41 | 112.63 |

Fixed-target document reranking is **+8.39 F1 percentage points** over
size-matched top-k. Marginal-coverage document reranking is **-11.42 points**
versus its matched top-k baseline: precision rises, but recall falls sharply.
This shows the fixed-target F1 improvement is not only a consequence of
returning fewer words, but the newer marginal-coverage selection policy does
not improve this document-pattern KPI over size-matched top-k.

These are still coarse filename-pattern metrics, not passage-level relevance
or answer correctness. Similarity and traversal thresholds remain hand-set;
this comparison does not calibrate them.

## Matched-context LLM answer QA pilot

Seven assistant-authored question/fact/source-page labels in
[`evaluation/docugami_passage_labels.json`](../evaluation/docugami_passage_labels.json)
were used to compare top-k, fixed-target and marginal-coverage answers. For
each question, all three arms received exactly the same whitespace-word count.
GPT-4o-mini generated the answers and the same model judged them against the
required facts and labeled source pages. Citation identifiers and cited
retrieved passages were also provided to the judge; the application's strict
citation-assessment gate was not run. The report is produced by
[`evaluation/docugami_answer_qa.py`](../evaluation/docugami_answer_qa.py).

| Retrieval arm | Correctness (0–4) | Completeness (0–4) | Citation support (0–4) |
|---|---:|---:|---:|
| Top-k | 3.00 | 3.43 | 3.00 |
| Fixed target | 4.00 | 4.00 | 4.00 |
| Marginal coverage | 4.00 | 4.00 | 4.00 |

By summed rubric scores, fixed-target versus top-k had 3 wins, 0 losses and 4
ties; marginal-coverage versus top-k had the same; fixed-target versus
marginal-coverage tied on all 7 questions. There were no answer-generation or
citation-format failures in this run.

This is a small, model-judged pilot, not independent validation: labels were
checked against the cited PDF pages but were not independently reviewed, and
the answer and judge models are the same. It suggests answer-quality
improvement over top-k on this sample, but does not establish a general
improvement, statistically reliable effect, or the impact of `QUALIFIES`
edges. The questions are not a dedicated exception-query test set, and the
heuristic qualifier extractor needs separate evaluation. See the
[passage-level QA review](docugami-passage-review.md) for prior retrieval-only
checks and limitations.

## Reproduce

Download the upstream PDFs and SEC `qna_data.csv` into
`testfiles/kg-rag-datasets`, preserving their upstream directory structure.
Keep the upstream license with the download. To reproduce the file manifest
above, remove `NCT06155240_Prot_SAP_000.pdf` from the local test corpus; the
script otherwise reports it as unextractable. The script creates or reuses a
separate database, skips unreadable PDFs with an explicit error, and excludes
questions whose labeled source files could not be indexed:

```powershell
py -3.14 -m pip install -e ".[evaluation]"
py -3.14 evaluation\docugami_benchmark.py `
  --data-dir testfiles\kg-rag-datasets `
  --database testfiles\kg-rag-datasets\docugami-minilm.db `
  --report testfiles\kg-rag-datasets\benchmark-results-current-legacy.json `
  --comparators --legacy-chunking
```

The first run downloads the pretrained models and may take considerable time and
local CPU and disk resources. The JSON report records every indexed file, each
excluded question and every baseline's per-question results. This benchmark
uses an optional local model adapter only for evaluation; it does not change the
application's selected embedding provider. The benchmark accepts either
`--document-selection-policy marginal_coverage` or
`--document-selection-policy fixed_target` to compare the current
coverage-driven default with the prior target-oriented policy. Its
size-matched top-k and document-reranking rows make the context comparison
within each policy explicit.

To evaluate the production LLM sector selector on the full-pipeline baseline,
add `--llm-sector-selection`. This makes one GPT-4o-mini routing request for
each eligible question; it requires `OPENAI_API_KEY` and incurs provider cost.
The query planner remains deterministic so the comparison isolates sector
selection. The report records whether selection was deterministic or LLM-based.

For the newer five-method comparison, omit `--legacy-chunking` and
`--comparators` and use a **different** database and report path, for example
`testfiles\kg-rag-datasets\docugami-minilm-token-aware.db` and
`testfiles\kg-rag-datasets\benchmark-results-token-aware.json`. Existing
indexes do not acquire the new chunk boundaries automatically.

## Indexed files

The following exact filenames were indexed in their upstream sector directories:

**SEC 10-Q (18):** `2022 Q3 AAPL.pdf`, `2022 Q3 AMZN.pdf`,
`2022 Q3 INTC.pdf`, `2022 Q3 MSFT.pdf`, `2022 Q3 NVDA.pdf`,
`2023 Q1 AAPL.pdf`, `2023 Q1 AMZN.pdf`, `2023 Q1 INTC.pdf`,
`2023 Q1 NVDA.pdf`, `2023 Q2 AAPL.pdf`, `2023 Q2 AMZN.pdf`,
`2023 Q2 INTC.pdf`, `2023 Q2 NVDA.pdf`, `2023 Q3 AAPL.pdf`,
`2023 Q3 AMZN.pdf`, `2023 Q3 INTC.pdf`, `2023 Q3 MSFT.pdf`,
`2023 Q3 NVDA.pdf`.

**Aviation (20):** `20071229X02007.pdf`, `20071231X02009.pdf`,
`20080102X00002.pdf`, `20080102X00005.pdf`, `20080104X00020.pdf`,
`20080108X00029.pdf`, `20080108X00031.pdf`, `20080109X00035.pdf`,
`20080111X00040.pdf`, `20080111X00041.pdf`, `20080114X00043.pdf`,
`20080115X00053.pdf`, `20080116X00055.pdf`, `20080116X00061.pdf`,
`20080117X00064.pdf`, `20080117X00072.pdf`, `20080122X00080.pdf`,
`20080123X00097.pdf`, `20080124X00100.pdf`, `20080124X00101.pdf`.

**Clinical trials (19):** `NCT06145295_Prot_SAP_000.pdf`,
`NCT06147739_Prot_SAP_000.pdf`, `NCT06149832_Prot_000.pdf`,
`NCT06150378_Prot_SAP_000.pdf`, `NCT06151600_Prot_000.pdf`,
`NCT06152367_Prot_000.pdf`, `NCT06152666_Prot_000.pdf`,
`NCT06152679_Prot_000.pdf`, `NCT06152900_Prot_SAP_000.pdf`,
`NCT06152952_Prot_000.pdf`, `NCT06153992_ICF_001.pdf`,
`NCT06153992_Prot_SAP_000.pdf`, `NCT06154122_Prot_000.pdf`,
`NCT06155006_Prot_SAP_000.pdf`, `NCT06155955_Prot_SAP_000.pdf`,
`NCT06156735_Prot_SAP_000.pdf`, `NCT06157684_Prot_000.pdf`,
`NCT06158269_Prot_000.pdf`, `NCT06159946_Prot_000.pdf`.

**Federal reports (20):** `2013-annual-performance-report.pdf`,
`2014-annual-performance-report.pdf`, `2015-annual-performance-report.pdf`,
`2016-annual-performance-report.pdf`, `2017-annual-performance-report.pdf`,
`2018-annual-performance-report.pdf`, `2019-annual-performance-report.pdf`,
`2020-annual-performance-report.pdf`, `2021-annual-performance-report.pdf`,
`2022-annual-performance-report.pdf`, `DHS2020.pdf`, `DOC2022.pdf`,
`DOL2022.pdf`, `DOSAID2022.pdf`, `DOT2022.pdf`, `EDAPP2022.pdf`,
`EPA2022.pdf`, `GSA2022.pdf`, `HHS2022.pdf`, `NARA2022.pdf`.

The two skipped SEC files were `2023 Q1 MSFT.pdf` and `2023 Q2 MSFT.pdf`;
the removed scanned file was `NCT06155240_Prot_SAP_000.pdf`.
