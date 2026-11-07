"""Telugu-aware lexical sparse vectors for hybrid search (no model, no network).

Features per text:
  - normalised word tokens (Telugu + Latin, digits kept: "2026", "6-08")
  - akshara bigrams inside each Telugu word ("కన్యా|రా", "రా|శి") - these make matching
    tolerant to OCR variants and inflection ("కన్యారాశి" vs "కన్యా రాశి వారికి")
Document weights are 1 + ln(tf); query terms weigh 1. Qdrant's IDF modifier on the sparse
vector supplies the inverse document frequency, giving BM25-like ranking.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Sequence

import regex

from panchangasetu.core.interfaces import SparseVector
from panchangasetu.telugu.text import graphemes, normalize_telugu

_TOKEN_RE = regex.compile(r"[\p{Telugu}]+|[a-z]+|\d+(?:-\d+)*")
_TELUGU_RE = regex.compile(r"\p{Telugu}")
_STOP = frozenset(
    {"the", "a", "an", "of", "is", "are", "what", "when", "how", "in", "for", "and", "to", "this", "be"}
)


def _hash(feature: str) -> int:
    return (
        int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=4).digest(), "big") & 0x7FFFFFFF
    )


def features(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(normalize_telugu(text).lower()):
        if tok in _STOP:
            continue
        out.append(f"w:{tok}")
        if _TELUGU_RE.match(tok):
            gs = graphemes(tok)
            out.extend(f"b:{a}{b}" for a, b in itertools.pairwise(gs))
    return out


class TeluguSparseEncoder:
    name = "telugu-lexical-v1"

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        vectors: list[SparseVector] = []
        for text in texts:
            counts = Counter(_hash(f) for f in features(text))
            idx = sorted(counts)
            vectors.append((idx, [1.0 + math.log(counts[i]) for i in idx]))
        return vectors

    def encode_query(self, text: str) -> SparseVector:
        idx = sorted({_hash(f) for f in features(text)})
        return idx, [1.0] * len(idx)
