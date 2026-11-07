"""Structure-aware chunking for Telugu documents.

Steps:
  1. Remove running headers/footers (lines that recur on many pages of the same document,
     e.g. "31 శ్రీ పరాభవనామ సంవత్సర సిద్ధాన్త పంచాంగం") so they don't pollute every chunk.
  2. Walk pages in order, splitting into blocks (blank-line separated). `## ` lines are
     headings: they close the current section and become the chunk's heading path.
  3. Pack blocks into chunks up to a token target; oversized blocks are split on sentence /
     danda boundaries, table-like blocks on row boundaries, and as a last resort on grapheme
     clusters - never inside a Telugu syllable.
  4. A collection change always closes a chunk, so a chunk never mixes collections.

Token counts are approximate until the embedder's tokenizer is pinned (Phase 3); the counter
is injectable for that reason.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import regex

from panchangasetu.core.types import Chunk, Collection
from panchangasetu.telugu.text import graphemes, split_sentences

TokenCounter = Callable[[str], int]

_DIGITS_RE = regex.compile(r"\d+")
# A heading stays in force for its own page and at most this many following pages.
_HEADING_TTL_PAGES = 2
_HEADING_PREFIX = "## "


def approx_token_counter(chars_per_token: float) -> TokenCounter:
    def count(text: str) -> int:
        return math.ceil(len(graphemes(text)) / chars_per_token) if text else 0

    return count


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int
    text: str
    collection: Collection
    section_type: str


def _boilerplate_key(line: str) -> str:
    return _DIGITS_RE.sub("#", line.strip().removeprefix(_HEADING_PREFIX)).strip()


_LETTER_RE = regex.compile(r"\p{L}")
# A boilerplate candidate must carry real words. Short recurring lines such as "6-20" (sunrise)
# or "భో.#-# అహః #-#" (daily durations) are table data and must survive.
_MIN_BOILERPLATE_LETTERS = 10


def _has_letters(text: str, minimum: int = 1) -> bool:
    return len(_LETTER_RE.findall(text)) >= minimum


def find_boilerplate_lines(
    pages: Sequence[PageText], min_share: float = 0.15, min_pages: int = 3
) -> set[str]:
    """Lines (digits masked) recurring across many pages are running headers/footers.

    The share is deliberately low: books alternate recto/verso headers, so each variant
    appears on at most ~50% of pages, and OCR variance splits that further.
    """
    if len(pages) < min_pages:
        return set()
    seen: Counter[str] = Counter()
    for page in pages:
        keys = {_boilerplate_key(line) for line in page.text.splitlines() if line.strip()}
        seen.update(k for k in keys if len(k) <= 160 and _has_letters(k, _MIN_BOILERPLATE_LETTERS))
    threshold = max(min_pages, math.ceil(min_share * len(pages)))
    return {k for k, n in seen.items() if n >= threshold}


def _letters_key(line: str) -> str:
    return "".join(_LETTER_RE.findall(line))


def _edit_distance_capped(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def is_boilerplate(line: str, boilerplate: set[str], letter_keys: list[str] | None = None) -> bool:
    """Exact (digits masked) match, or a near-miss spelling of a running header: OCR and the
    typesetter vary it ("సిద్ధాంత" vs "సిద్ధాన్త"), and the rare variant never reaches the
    frequency threshold on its own."""
    if _boilerplate_key(line) in boilerplate:
        return True
    letters = _letters_key(line)
    if len(letters) < _MIN_BOILERPLATE_LETTERS:
        return False
    keys = letter_keys if letter_keys is not None else [_letters_key(k) for k in boilerplate]
    return any(_edit_distance_capped(letters, k, 2) <= 2 for k in keys)


def strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    if not boilerplate:
        return text
    keys = [_letters_key(k) for k in boilerplate]
    return "\n".join(line for line in text.splitlines() if not is_boilerplate(line, boilerplate, keys))


def _is_table_like(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    return len(lines) >= 2 and sum(" | " in ln for ln in lines) >= len(lines) / 2


def _hard_split(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Last resort: split on whitespace, then on grapheme clusters, never mid-syllable."""
    pieces: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if count(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if count(word) <= max_tokens:
            current = word
            continue
        buf = ""
        for g in graphemes(word):
            if count(buf + g) > max_tokens and buf:
                pieces.append(buf)
                buf = ""
            buf += g
        current = buf
    if current:
        pieces.append(current)
    return pieces


def split_block(block: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split one block into units that each fit in max_tokens."""
    if count(block) <= max_tokens:
        return [block]
    units = (
        [ln for ln in block.splitlines() if ln.strip()]
        if _is_table_like(block) or block.count("\n") >= 3
        else split_sentences(block)
    )
    out: list[str] = []
    for unit in units:
        out.extend([unit] if count(unit) <= max_tokens else _hard_split(unit, max_tokens, count))
    return out


@dataclass(slots=True)
class _Open:
    units: list[str]
    collection: Collection
    section_type: str
    heading_path: list[str]
    page_start: int
    page_end: int


class TeluguChunker:
    def __init__(
        self,
        *,
        target_tokens: int,
        max_tokens: int,
        overlap_units: int,
        count_tokens: TokenCounter,
        min_tokens: int = 30,
    ) -> None:
        if not min_tokens <= target_tokens <= max_tokens:
            raise ValueError("require min_tokens <= target_tokens <= max_tokens")
        self.min = min_tokens
        self.target = target_tokens
        self.max = max_tokens
        self.overlap = overlap_units
        self.count = count_tokens

    def chunk(self, pages: Sequence[PageText]) -> list[Chunk]:
        boilerplate = find_boilerplate_lines(pages)
        chunks: list[Chunk] = []
        current: _Open | None = None
        headings: list[str] = []
        heading_page = 0
        last_section: tuple[Collection, str] | None = None

        def flush(carry_overlap: bool) -> None:
            nonlocal current
            if current is None or not current.units:
                current = None
                return
            text = "\n\n".join(current.units)
            chunks.append(
                Chunk(
                    text=text,
                    collection=current.collection,
                    section_type=current.section_type,
                    heading_path=list(current.heading_path),
                    page_start=current.page_start,
                    page_end=current.page_end,
                    chunk_index=len(chunks),
                    token_count=self.count(text),
                )
            )
            if carry_overlap and self.overlap:
                tail = current.units[-self.overlap :]
                if self.count("\n\n".join(tail)) < self.target // 2:
                    current = _Open(
                        list(tail),
                        current.collection,
                        current.section_type,
                        current.heading_path,
                        current.page_end,
                        current.page_end,
                    )
                    return
            current = None

        for page in pages:
            if page.collection == Collection.SKIP:
                flush(carry_overlap=False)
                continue
            text = strip_boilerplate(page.text, boilerplate)
            # Headings describe the section they open. They must not leak across a curated
            # section change, nor live on through pages that carry no headings of their own.
            section = (page.collection, page.section_type)
            if section != last_section or page.page_number - heading_page > _HEADING_TTL_PAGES:
                headings = []
            last_section = section
            for raw_block in regex.split(r"\n\s*\n", text):
                block = raw_block.strip()
                if not block:
                    continue
                if block.startswith(_HEADING_PREFIX):
                    lines = [ln.removeprefix(_HEADING_PREFIX).strip() for ln in block.splitlines()]
                    titled = [ln for ln in lines if _has_letters(ln)]
                    if titled:  # "## 156" (a page number tagged as header) is not a heading
                        flush(carry_overlap=False)
                        # A header block opens with its title(s) - sometimes a section title and a
                        # sub-title ("…రాశి ఫలితములు" then "మేషరాశి ఫలితములు") - and ends with details
                        # (nakshatra padas, aaya/vyaya). Keep the first two lines and the last.
                        headings = list(dict.fromkeys(titled[:2] + titled[-1:]))
                        heading_page = page.page_number
                    continue
                if current is not None and (
                    current.collection != page.collection or current.section_type != page.section_type
                ):
                    flush(carry_overlap=False)
                for unit in split_block(block, self.max, self.count):
                    if current is None:
                        current = _Open(
                            [],
                            page.collection,
                            page.section_type,
                            list(headings),
                            page.page_number,
                            page.page_number,
                        )
                    candidate = "\n\n".join([*current.units, unit])
                    if current.units and self.count(candidate) > self.target:
                        flush(carry_overlap=True)
                        if current is not None and self.count("\n\n".join([*current.units, unit])) > self.max:
                            current.units = []  # overlap would push this chunk past max_tokens
                        if current is None:
                            current = _Open(
                                [],
                                page.collection,
                                page.section_type,
                                list(headings),
                                page.page_number,
                                page.page_number,
                            )
                    current.units.append(unit)
                    current.page_end = page.page_number
        flush(carry_overlap=False)
        return self._merge_small(chunks)

    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fold chunks below min_tokens (stray title lines, page-end fragments) into a neighbour
        of the same collection and section, preferring the following chunk: a title line
        belongs with the content it introduces."""

        def joinable(a: Chunk, b: Chunk) -> bool:
            return (
                a.collection == b.collection
                and a.section_type == b.section_type
                and self.count(a.text + "\n\n" + b.text) <= self.max
            )

        out: list[Chunk] = []
        pending: Chunk | None = None  # small chunk waiting to be prepended to the next one
        for ch in chunks:
            if pending is not None:
                if joinable(pending, ch):
                    ch = Chunk(
                        text=pending.text + "\n\n" + ch.text,
                        collection=ch.collection,
                        section_type=ch.section_type,
                        heading_path=ch.heading_path or pending.heading_path,
                        page_start=min(pending.page_start, ch.page_start),
                        page_end=max(pending.page_end, ch.page_end),
                        chunk_index=0,
                        token_count=self.count(pending.text + "\n\n" + ch.text),
                        metadata=ch.metadata,
                    )
                elif out and joinable(out[-1], pending):
                    out[-1] = self._append(out[-1], pending)
                else:
                    out.append(pending)
                pending = None
            if ch.token_count < self.min:
                pending = ch
            else:
                out.append(ch)
        if pending is not None:
            if out and joinable(out[-1], pending):
                out[-1] = self._append(out[-1], pending)
            else:
                out.append(pending)
        for i, ch in enumerate(out):
            ch.chunk_index = i
        return out

    def _append(self, a: Chunk, b: Chunk) -> Chunk:
        text = a.text + "\n\n" + b.text
        return Chunk(
            text=text,
            collection=a.collection,
            section_type=a.section_type,
            heading_path=a.heading_path,
            page_start=min(a.page_start, b.page_start),
            page_end=max(a.page_end, b.page_end),
            chunk_index=a.chunk_index,
            token_count=self.count(text),
            metadata=a.metadata,
        )
