"""Vendored BM25Okapi implementation compatible with rank_bm25 0.2.2.

The production retrieval path scores corpus chunks with
``rank_bm25.BM25Okapi`` over ``re.findall(r"[\\w']+", text.casefold())`` tokens.
That package is not installed in the local evaluation environment, so this
module carries a faithful port of its scoring math (k1=1.5, b=0.75,
epsilon=0.25, negative-IDF flooring by ``epsilon * average_idf``).

Source: rank_bm25 0.2.2 (MIT License, https://github.com/dorianbrown/rank_bm25).
Only the scoring surface is ported; corpus tokenization and ranking order
remain the caller's responsibility, mirroring the production call sites.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[\w']+")


def tokenize(text: object) -> list[str]:
    return _TOKEN_RE.findall(str(text).casefold())


class BM25Okapi:
    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus: Sequence[Sequence[str]]) -> dict[str, int]:
        nd: dict[str, int] = {}
        for document in corpus:
            self.doc_len.append(len(document))
            self.corpus_size += 1
            frequencies: dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        return nd

    def _calc_idf(self, nd: dict[str, int]) -> None:
        idf_sum = 0.0
        negative_idfs: list[str] = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: Iterable[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size)
        doc_len = np.asarray(self.doc_len, dtype=float)
        if not self.corpus_size or not doc_len.any():
            return score
        for term in query:
            q_freq = np.asarray([doc.get(term, 0) for doc in self.doc_freqs], dtype=float)
            score += (self.idf.get(term, 0.0)) * (
                q_freq * (self.k1 + 1.0)
                / (q_freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl))
            )
        return score
