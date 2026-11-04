"""Deterministic quote location for quote-first claim extraction.

The extraction model returns only a verbatim ``quote``. This module locates the
quote inside the owning passage so claim spans are exact by construction:

    chunk_text[start:end] == quote            (method "exact")
    normalize(chunk_text[start:end]) == normalize(quote)   (method "normalized")
    best sentence/window aligned to the quote (method "sentence_window")

Pure stdlib (difflib). An optional rapidfuzz import is guarded and only used
when a caller explicitly opts in; the default path is fully deterministic.

No model inference, no network.
"""

from __future__ import annotations

import sys
import unicodedata
from difflib import SequenceMatcher

try:  # optional accelerator; never required and never used by default
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except Exception:  # pragma: no cover - environment dependent
    _rapidfuzz_fuzz = None

# Typographic folds applied after NFKC (NFKC does not fold curly quotes).
_CURLY_FOLD = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u00ab": '"', "\u00bb": '"', "\u2033": '"',
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'",
    "\u2039": "'", "\u203a": "'",
}
_ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\ufeff")

_SENTENCE_TERMINATORS = frozenset({".", "!", "?", "\u2026", "\u0964", "\u0965"})
_CLOSERS = frozenset({'"', "'", "\u201d", "\u2019", ")", "]", "}", "\u00bb", "\u203a"})


def normalize_with_index_map(text: str) -> tuple[str, list[int]]:
    """NFKC + curly-quote fold + casefold + whitespace collapse.

    Returns ``(normalized, index_map)`` where ``index_map[i]`` is the original
    character index that produced normalized character ``i``, and the final
    sentinel ``index_map[len(normalized)] == len(text)``. Characters that
    expand under NFKC/casefold (``ß`` -> ``ss``, ``İ`` -> ``i\u0307``) map every
    expanded character back to the same source index. Whitespace runs collapse
    to one space mapped to the first whitespace character of the run.
    Zero-width characters are dropped.
    """

    out: list[str] = []
    index_map: list[int] = []
    pending_space = False
    pending_space_index = -1

    for i, ch in enumerate(text):
        segment = unicodedata.normalize("NFKC", ch)
        for zw in _ZERO_WIDTH:
            segment = segment.replace(zw, "")
        segment = "".join(_CURLY_FOLD.get(c, c) for c in segment)
        segment = segment.casefold()
        if not segment:
            continue
        if segment.strip() == "":
            if not pending_space:
                pending_space = True
                pending_space_index = i
            continue
        if pending_space:
            if out:
                out.append(" ")
                index_map.append(pending_space_index)
            pending_space = False
    # note: leading and trailing whitespace emit nothing
        for c in segment:
            out.append(c)
            index_map.append(i)

    normalized = "".join(out)
    index_map.append(len(text))
    return normalized, index_map


def sentence_split(text: str) -> list[tuple[int, int]]:
    """Split into sentence spans ``(start, end)`` (whitespace trimmed).

    Boundaries: ``.`` ``!`` ``?`` ``\u2026`` (ellipsis) and Devanagari danda
    ``\u0964`` ``\u0965``. A terminator ends a sentence only when followed by
    whitespace/end (so ``U.S. officials`` is not split). Paragraph breaks
    (two or more newlines) also end a sentence.
    """

    if not isinstance(text, str) or not text:
        return []
    spans: list[tuple[int, int]] = []
    n = len(text)
    start = 0
    i = 0

    def emit(s: int, e: int) -> None:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))

    while i < n:
        ch = text[i]
        if ch == "\n":
            j = i
            while j < n and text[j] == "\n":
                j += 1
            if j - i >= 2:
                emit(start, i)
                start = j
                i = j
                continue
            i += 1
            continue
        if ch in _SENTENCE_TERMINATORS:
            j = i + 1
            while j < n and text[j] in _SENTENCE_TERMINATORS:
                j += 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            if j >= n or text[j].isspace():
                emit(start, j)
                start = j
                i = j
                continue
        i += 1
    emit(start, n)
    return spans


def _ratio(a: str, b: str, use_rapidfuzz: bool = False) -> float:
    if not a or not b:
        return 0.0
    if use_rapidfuzz and _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.ratio(a, b)) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def _find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    positions: list[int] = []
    pos = haystack.find(needle)
    while pos != -1:
        positions.append(pos)
        pos = haystack.find(needle, pos + 1)
    return positions


def _hint_center(hint) -> int | None:
    if hint is None:
        return None
    if isinstance(hint, bool):
        return None
    if isinstance(hint, int):
        return hint
    if isinstance(hint, (tuple, list)) and len(hint) == 2 and all(isinstance(v, int) for v in hint):
        return (hint[0] + hint[1]) // 2
    return None


def _pick_occurrence(positions: list[tuple[int, int]], hint) -> tuple[int, int]:
    center = _hint_center(hint)
    if center is None or len(positions) == 1:
        return positions[0]
    return min(positions, key=lambda se: (abs(se[0] - center), se[0]))


def locate_quote(
    chunk_text: str,
    quote: str,
    min_chars: int = 40,
    min_words: int = 5,
    hint=None,
    align_threshold: float = 0.90,
) -> dict | None:
    """Locate ``quote`` inside ``chunk_text``.

    Order: exact -> normalized-with-index-map -> sentence/window alignment.
    Returns ``{"start", "end", "method", "score", "quote"}`` or ``None``.
    The returned ``quote`` is always ``chunk_text[start:end]``.

    ``hint`` disambiguates duplicate occurrences: an int (legacy span start) or
    an ``(start, end)`` tuple (e.g. a candidate mention span).
    """

    if not isinstance(chunk_text, str) or not isinstance(quote, str) or not chunk_text:
        return None
    q = quote.strip()
    if len(q) < min_chars or len(q.split()) < min_words:
        return None

    occurrences = [(p, p + len(q)) for p in _find_all(chunk_text, q)]
    if occurrences:
        start, end = _pick_occurrence(occurrences, hint)
        return {"start": start, "end": end, "method": "exact", "score": 1.0, "quote": chunk_text[start:end]}

    normalized_chunk, index_map = normalize_with_index_map(chunk_text)
    normalized_quote, _ = normalize_with_index_map(q)
    if normalized_quote:
        norm_occurrences = [
            (index_map[p], index_map[p + len(normalized_quote)])
            for p in _find_all(normalized_chunk, normalized_quote)
        ]
        if norm_occurrences:
            start, end = _pick_occurrence(norm_occurrences, hint)
            return {
                "start": start,
                "end": end,
                "method": "normalized",
                "score": 1.0,
                "quote": chunk_text[start:end],
            }

    aligned = align_paraphrase_to_quote(chunk_text, q, anchors=None, threshold=align_threshold)
    if aligned is not None:
        return aligned
    return None


def align_paraphrase_to_quote(
    chunk_text: str,
    paraphrase: str,
    anchors=None,
    threshold: float = 0.75,
    use_rapidfuzz: bool = False,
) -> dict | None:
    """Best sentence/window quote for a paraphrase (legacy repair path).

    Candidate windows: single sentences and runs of 2-3 adjacent sentences
    (``sentence_split``). ``anchors`` is an optional list of strings that must
    each appear in the candidate window (normalized) for the window to qualify.
    Score = similarity over normalized text; best window at/above ``threshold``
    is returned as ``{"start", "end", "method": "sentence_window", "score",
    "quote"}``, else ``None``.
    """

    if not isinstance(chunk_text, str) or not isinstance(paraphrase, str) or not chunk_text:
        return None
    sentences = sentence_split(chunk_text)
    if not sentences:
        return None
    norm_paraphrase, _ = normalize_with_index_map(paraphrase)
    if not norm_paraphrase:
        return None

    normalized_anchors: list[str] = []
    for anchor in anchors or []:
        if isinstance(anchor, str) and anchor.strip():
            norm_anchor, _ = normalize_with_index_map(anchor)
            if norm_anchor:
                normalized_anchors.append(norm_anchor)

    best = None  # (score, span_len, start, end)
    for width in (1, 2, 3):
        for i in range(0, len(sentences) - width + 1):
            start = sentences[i][0]
            end = sentences[i + width - 1][1]
            window = chunk_text[start:end]
            norm_window, _ = normalize_with_index_map(window)
            if not norm_window:
                continue
            if normalized_anchors and not all(a in norm_window for a in normalized_anchors):
                continue
            score = _ratio(norm_paraphrase, norm_window, use_rapidfuzz=use_rapidfuzz)
            candidate = (score, end - start, start, end)
            if best is None or candidate > best:
                best = candidate

    if best is None or best[0] < threshold:
        return None
    score, _, start, end = best
    return {
        "start": start,
        "end": end,
        "method": "sentence_window",
        "score": score,
        "quote": chunk_text[start:end],
    }


def main() -> int:
    chunk = (
        "The committee reported that housing discrimination remains widespread "
        "in many regions. \u201cAccess to shelter should not depend on who you "
        "are,\u201d the report states. Hate speech incidents rose sharply in the "
        "same period, according to municipal records."
    )

    print("== exact ==")
    print(locate_quote(chunk, "Hate speech incidents rose sharply in the same period"))

    print("\n== normalized (curly quotes, NBSP, case) ==")
    print(locate_quote(chunk, '"Access to shelter should not depend on who you are," the report states.'))

    print("\n== sentence/window alignment on a paraphrase ==")
    print(align_paraphrase_to_quote(chunk, "The committee said that housing discrimination remains widespread in many regions of the country"))

    print("\n== duplicate occurrences, nearest to hint ==")
    repeated = "Alpha beta gamma delta epsilon zeta eta theta " * 2
    print(locate_quote(repeated, "Alpha beta gamma delta epsilon zeta eta theta", hint=60))

    print("\n== too short -> None ==")
    print(locate_quote(chunk, "Hate speech"))

    print("\n== unrelated quote -> None ==")
    print(locate_quote(chunk, "The weather forecast predicts heavy rain across the northern provinces tomorrow morning"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
