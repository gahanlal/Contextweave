"""Additional local retrieval baselines for the Docugami experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from time import perf_counter

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from isthmus.evaluation.baselines import BaselineResult
from isthmus.query import terms


MODEL_REPOSITORY = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class HybridSearch:
    def __init__(self, chunks):
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.postings = defaultdict(list)
        self.lengths = {}
        for chunk in self.chunks.values():
            # Use word occurrences rather than the set returned by terms().
            words = [word for word in chunk.content.lower().split() if word]
            frequencies = Counter(token for word in words for token in terms(word))
            self.lengths[chunk.id] = sum(frequencies.values())
            for token, count in frequencies.items():
                self.postings[token].append((chunk.id, count))
        self.average_length = sum(self.lengths.values()) / max(1, len(self.lengths))

    def rank(self, query, dense_hits, limit):
        scores = defaultdict(float)
        count = len(self.chunks)
        for token in terms(query):
            postings = self.postings.get(token, ())
            if not postings:
                continue
            idf = log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for chunk_id, frequency in postings:
                length = self.lengths[chunk_id] / max(self.average_length, 1)
                scores[chunk_id] += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length))
        lexical = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
        fused = defaultdict(float)
        for rank, chunk_id in enumerate(lexical, start=1):
            fused[chunk_id] += 1 / (60 + rank)
        for rank, hit in enumerate(dense_hits, start=1):
            fused[hit.chunk.id] += 1 / (60 + rank)
        return tuple(self.chunks[chunk_id] for chunk_id in sorted(fused, key=lambda item: (-fused[item], item))[:limit])


class CrossEncoder:
    def __init__(self):
        tokenizer_path = hf_hub_download(MODEL_REPOSITORY, "tokenizer.json", revision=MODEL_REVISION)
        model_path = hf_hub_download(MODEL_REPOSITORY, "onnx/model_quint8_avx2.onnx", revision=MODEL_REVISION)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])

    def rerank(self, query, chunks, limit=10):
        if not chunks:
            return ()
        encodings = self.tokenizer.encode_batch([(query, chunk.content) for chunk in chunks])
        features = {
            "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
            "attention_mask": np.asarray([item.attention_mask for item in encodings], dtype=np.int64),
            "token_type_ids": np.asarray([item.type_ids for item in encodings], dtype=np.int64),
        }
        inputs = {item.name: features[item.name] for item in self.session.get_inputs()}
        scores = self.session.run(None, inputs)[0].reshape(-1)
        if len(scores) != len(chunks) or not np.isfinite(scores).all():
            raise ValueError("Cross-encoder returned invalid scores.")
        return tuple(
            chunks[position]
            for position in sorted(range(len(chunks)), key=lambda position: (-float(scores[position]), chunks[position].id))[:limit]
        )


class PageMerger:
    def __init__(self, chunks):
        self.pages = defaultdict(list)
        for chunk in chunks:
            self.pages[(chunk.document_id, chunk.source_location)].append(chunk)
        for page in self.pages.values():
            page.sort(key=lambda chunk: chunk.sequence_number)

    def merge(self, hits):
        hit_ids = {hit.chunk.id for hit in hits}
        seen = set()
        result = []
        for hit in hits:
            chunk = hit.chunk
            page = self.pages[(chunk.document_id, chunk.source_location)]
            count = sum(item.id in hit_ids for item in page)
            items = page if count >= 2 and count / len(page) >= 0.5 else (chunk,)
            for item in items:
                if item.id not in seen:
                    result.append(item)
                    seen.add(item.id)
        return tuple(result)


def run_comparators(index, query, hybrid, cross_encoder, merger, limit):
    started = perf_counter()
    vector = index.embed_query(query)
    hits = index.search_by_vector(vector, limit=limit)
    search_ms = (perf_counter() - started) * 1000

    started = perf_counter()
    hybrid_chunks = hybrid.rank(query, hits, limit)
    hybrid_ms = search_ms + (perf_counter() - started) * 1000

    started = perf_counter()
    reranked = cross_encoder.rerank(query, hybrid_chunks)
    rerank_ms = hybrid_ms + (perf_counter() - started) * 1000

    started = perf_counter()
    merged = merger.merge(hits)
    merge_ms = search_ms + (perf_counter() - started) * 1000
    return tuple(
        BaselineResult(name, chunks, tuple(dict.fromkeys(chunk.document_id for chunk in chunks)), latency)
        for name, chunks, latency in (
            ("F: hybrid BM25 + vector", hybrid_chunks, hybrid_ms),
            ("G: hybrid + cross-encoder", reranked, rerank_ms),
            ("H: page auto-merge", merged, merge_ms),
        )
    )
