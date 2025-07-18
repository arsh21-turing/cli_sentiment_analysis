"""Utility helpers for splitting large texts into manageable chunks while preserving sentence
boundaries and for merging per-chunk sentiment/emotion analysis back into a single result.

The heuristics are deliberately simple so the function has **zero external
runtime dependencies** – it relies only on the Python std-lib (``re`` & ``math``).
"""
from __future__ import annotations

import re
from typing import List, Dict, Any
# ---------------------------------------------------------------------------
# Optional progress-bar dependency
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import types, sys

    def tqdm(iterable, *_, **__):  # type: ignore
        return iterable

    dummy = types.ModuleType("tqdm")
    dummy.tqdm = tqdm  # type: ignore[attr-defined]
    sys.modules.setdefault("tqdm", dummy)

# Public API ---------------------------------------------------------------
__all__ = [
    "split_text",
    "process_chunked_text",
    "_combine_chunk_results",
]


# sentence boundary split – keep leading whitespace of following sentence
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?=\s)")


def split_text(text: str, *, max_chunk_size: int = 5_000, overlap: int = 0) -> List[str]:
    """Split *text* into chunks of at most *max_chunk_size* characters.

    The algorithm tries to respect sentence boundaries.  If a single sentence
    exceeds *max_chunk_size* it is hard-split.
    """
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [text]

    sentences = _SENTENCE_SPLIT_RE.split(text)
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""

    for sent in sentences:
        # If sentence fits in current chunk, append
        if len(current) + len(sent) <= max_chunk_size:
            current += sent
            continue

        # flush current and start new
        if current:
            chunks.append(current)
            current = ""

        # sentence itself longer than limit – hard split
        while len(sent) > max_chunk_size:
            chunks.append(sent[:max_chunk_size])
            sent = sent[max_chunk_size:]
        current = sent

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        ov_chunks: list[str] = []
        for i, ch in enumerate(chunks):
            if i < len(chunks) - 1:
                ov = chunks[i + 1][:overlap]
                ov_chunks.append(ch + ov)
            else:
                ov_chunks.append(ch)
        return ov_chunks
    return chunks


def process_chunked_text(
    text: str,
    analyzer,
    *,
    max_chunk_size: int = 5_000,
    overlap: int = 0,
):
    """Analyze *text* by splitting it into chunks, analysing each chunk and
    merging the predictions.

    The *analyzer* object must expose an ``analyze(str) -> dict`` API identical
    to the existing models in this project.
    """
    if not text:
        return {"error": "empty text", "chunked": False}

    if len(text) <= max_chunk_size:
        # small enough – use normal analysis path
        return analyzer.analyze(text)

    chunks = split_text(text, max_chunk_size=max_chunk_size, overlap=overlap)

    per_chunk: list[dict[str, Any]] = []
    for idx, ch in enumerate(tqdm(chunks, desc="Analyzing chunks", unit="chunk")):
        entry: Dict[str, Any] = {"chunk_index": idx, "chunk_size": len(ch)}
        try:
            entry["result"] = analyzer.analyze(ch)
        except Exception as exc:  # pragma: no cover
            entry["error"] = str(exc)
        per_chunk.append(entry)

    combined = _combine(per_chunk)
    return combined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _combine(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Weighted merge of chunk-level predictions."""
    valid = [c for c in chunks if "result" in c]
    total_chunks = len(chunks)
    successful = len(valid)

    if not valid:
        return {
            "error": "All chunks failed analysis",
            "chunked": True,
            "total_chunks": total_chunks,
            "successful_chunks": 0,
        }

    total = sum(c.get("chunk_size") or c.get("size") for c in valid)

    def _weighted_scores(label_key: str):
        agg: dict[str, float] = {}
        for c in valid:
            res = c["result"].get(label_key)
            if not res:
                continue
            size = c.get("chunk_size") or c.get("size")
            weight = size / total if total else 0

            # Use raw probabilities if present, else fall back to single score
            raw_probs = res.get("raw_probabilities")
            if raw_probs:
                for lbl, prob in raw_probs.items():
                    agg[lbl] = agg.get(lbl, 0.0) + prob * weight
            else:
                agg[res["label"]] = agg.get(res["label"], 0.0) + res.get("score", 0.0) * weight

        if not agg:
            return {"label": "unknown", "score": 0.0}  # Default to unknown if no scores
        top_label, top_score = max(agg.items(), key=lambda kv: kv[1])
        return {"label": top_label, "score": top_score}

    combined: Dict[str, Any] = {"chunked": True, "total_chunks": total_chunks, "successful_chunks": successful}
    sent = _weighted_scores("sentiment")
    if sent:
        combined["sentiment"] = sent
    emo = _weighted_scores("emotion")
    if emo:
        combined["emotion"] = emo
    return combined


# Export alias expected by unit-tests
_combine_chunk_results = _combine 