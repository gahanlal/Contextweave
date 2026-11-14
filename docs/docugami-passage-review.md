# Passage-level labels and answer QA

The current answer-quality pilot is the
[matched-context LLM QA evaluation](docugami-evaluation.md#matched-context-llm-answer-qa-pilot).
The retrieval-only checks below predate that pilot and use natural, unequal
context sizes; they are retained as historical observations, not controlled
evidence that one method is better.

## Historical retrieval-only checks (unequal contexts)

### Method

Seven questions were written from the source PDFs before inspecting retrieval
results. The expected answers and physical PDF page numbers below were checked
against the documents. The assistant then inspected the actual returned
passages for each answer component; a matching filename or page alone did not
count. Labels and judgments were **not independently human-reviewed**.

Both methods searched the complete 77-document original MiniLM index described
in the [experiment report](docugami-evaluation.md), using the current selection
code, deterministic planner and 4,500-word maximum estimated budget. This is the
**legacy word-chunked index**, not the later token-aware index. Top-k and the
full document-reranked pipeline received the same queries, but their returned
contexts were not truncated to equal sizes. An answer is complete only when
all requested facts are supported somewhere in the returned context; partial
means at least one requested fact is missing.

## Questions, answers and observations

Financial amounts are in millions of US dollars. Dates refer to the specific
three-month periods in the filings, which do not end on the same day.

| Question | Expected evidence, physical PDF pages | Top-k | Isthmus |
|---|---|---|---|
| In the 2023 Q3 Apple filing, what were net sales and net income for the three months ended July 1, 2023? | `2023 Q3 AAPL.pdf`, p. 4: 81,797 and 19,881. Other supporting passages: pp. 9–10. | Complete | Complete, using pp. 9–10 |
| In the 2023 Q3 Amazon filing, what were net sales and operating income for the three months ended September 30, 2023? | `2023 Q3 AMZN.pdf`, p. 4: 143,083 and 11,188. | Complete | Complete |
| Compare the three-month net sales reported in the filings labeled 2023 Q3 for Apple and Amazon. | `2023 Q3 AAPL.pdf`, pp. 4 or 10: 81,797; `2023 Q3 AMZN.pdf`, p. 4: 143,083. The periods end July 1 and September 30, respectively. | Partial: Apple only | Partial: Apple only |
| What did the NTSB determine was the probable cause of the Piper PA-22-150 accident near Tallahassee? | `20071229X02007.pdf`, p. 1: the pilot misjudged obstacle clearance on final approach. | Complete | Complete |
| What visibility and ceiling were reported for the Reims F172M accident near Chevroux, France? | `20080102X00002.pdf`, p. 3: 0.7-mile visibility and a broken ceiling at 200 feet above ground. | Complete | Complete |
| How many participants were planned for the single-center oral chronic graft-versus-host disease study using mesenchymal stem cell dressings? | `NCT06149832_Prot_000.pdf`, p. 3: 38 patients. | Complete | Complete |
| For the spinal-injury virtual-reality rehabilitation feasibility study, what was the intended intervention schedule and target recruitment? | `NCT06154122_Prot_000.pdf`, pp. 2–3: up to 30 minutes at least three times weekly for 12 weeks; target 24 participants. | Complete | Partial: schedule, but not the recruitment passage |

Top-k supplied complete evidence for **6/7** questions; Isthmus supplied
complete evidence for **5/7**. Both missed Amazon's necessary sales figure in
the comparison, even though top-k returned some Amazon material. For the VR
question, the reranked context included a chunk from the correct recruitment
page but not the chunk containing the recruitment figure. This is why page
matches are not evidence of answer support.

## Same questions on the token-aware index

The same seven questions were also run against the re-indexed corpus and
current pipeline. The assistant reviewed the returned passages again; these
are separate single-run judgments, not a controlled attribution of the effect
of chunking.

| Question | Top-k | Isthmus document reranking |
|---|---|---|
| Apple sales and net income | Complete: pp. 5 and 10 | Complete: pp. 9 and 10 |
| Amazon sales and operating income | Complete: p. 4 | Complete: p. 4 |
| Apple/Amazon sales comparison | Partial: Apple only | Complete: Apple p. 19, Amazon p. 24 |
| Piper probable cause | Not supported: the cause statement was not returned | Complete: p. 1 |
| Reims visibility and ceiling | Complete: p. 3 | Complete: p. 3 |
| Oral trial planned enrollment | Complete: p. 3 | Complete: p. 3 |
| VR schedule and target recruitment | Complete: pp. 2–3 | Partial: schedule only |

On this run top-k supplied complete evidence for **5/7** questions and
Isthmus for **6/7**. The multi-document comparison improved for the reranked
pipeline, while the top-k context omitted the Piper cause statement. The
reranked VR context still missed the recruitment figure.

The set is small, hand-selected and biased toward facts that could be verified
in the PDFs; it does not estimate general accuracy. It includes finance,
aviation and clinical-trial protocols but no federal-report question. No answer
model was run for these historical checks: they are **retrieval evidence
judgments**, not generated-answer quality, citation correctness, medical
conclusions or performance on independently held-out questions. Some PDF text
and tables are poorly extracted, and the original embedding model truncated
long chunks.

## Current matched-context answer QA pilot

The current seven-question labels are structured in
[`evaluation/docugami_passage_labels.json`](../evaluation/docugami_passage_labels.json).
They record required facts and evidence pages checked against the source PDFs.
Top-k, fixed-target and marginal-coverage answer contexts were truncated to
the same whitespace-word count per question before answer generation.

| Retrieval arm | Correctness (0–4) | Completeness (0–4) | Citation support (0–4) |
|---|---:|---:|---:|
| Top-k | 3.00 | 3.43 | 3.00 |
| Fixed target | 4.00 | 4.00 | 4.00 |
| Marginal coverage | 4.00 | 4.00 | 4.00 |

Fixed-target and marginal-coverage scores tied on all seven questions. Each
scored above top-k on three questions and tied on four. GPT-4o-mini generated
and judged answers; its scores are not independent human assessments. Labels
were assistant-authored and not independently reviewed, and the sample is too
small to establish a general answer-quality gain. It also contains no dedicated
exception-query set, so it does not measure the effect of `QUALIFIES` evidence
inclusion.

The API runner is
[`evaluation/docugami_answer_qa.py`](../evaluation/docugami_answer_qa.py). The
full methodology and document-pattern results are in the
[experiment report](docugami-evaluation.md).

## Reproduce the retrieval

Using the original local corpus/index and optional evaluation dependencies:

```powershell
py -3.14 evaluation\docugami_manual_qa.py `
  --database testfiles\kg-rag-datasets\docugami-minilm.db `
  --report testfiles\kg-rag-datasets\manual-passage-results.json `
  --legacy-chunking
```

The JSON contains the full returned contexts, source locations and retrieval
times; inspect their contents against the source PDFs rather than grading by
filename. It may contain document text and should not be published without
checking the source license. Approximate vector-search results and timings may
vary. To use the token-aware index, point to its database and omit
`--legacy-chunking`; that reproduces the token-aware experiment rather than
the legacy-index experiment. Results can vary between runs.

The reviewed local runs are saved in `testfiles\kg-rag-datasets` as
`manual-passage-results-legacy.json` and
`manual-passage-results-token-aware.json`. These local files are not included
in the repository.
