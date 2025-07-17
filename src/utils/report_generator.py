import os
import json
import csv
import shutil
from datetime import datetime
from typing import Dict, Any, Optional

# Optional terminal colours ---------------------------------------------------
try:
    from colorama import init, Fore, Style  # type: ignore

    init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:  # pragma: no cover – colour support is optional

    class _DummyColour:  # pylint: disable=too-few-public-methods
        """Fallback when colour library is missing."""

        def __getattr__(self, _name: str) -> str:  # noqa: D401
            return ""

    Fore = Style = _DummyColour()  # type: ignore
    COLOR_ENABLED = False

# ---------------------------------------------------------------------------

__all__ = [
    "BatchReportGenerator",
    "generate_batch_report",
    "export_batch_report",
]


class BatchReportGenerator:  # pylint: disable=too-many-instance-attributes
    """Create a human-readable summary for batch analysis results.

    The *results* object is expected to follow the structure returned by the
    existing ``process_batch`` helper in :pymod:`utils.batch`::

        {
            "total_texts": int,
            "source": str,
            "timestamp": str,  # ISO-like timestamp
            "statistics": {
                "sentiment_distribution": {"positive": 10, "neutral": 3, ...},
                "emotion_distribution":   {"joy": 4, "anger": 2, ...},
                "confidence_stats": {
                    "avg_confidence": float,
                    "min_confidence": float,
                    "max_confidence": float,
                    "std_dev": float,
                },
                "error_rate": float,  # 0-1
                "error_categories": {"APIError": 3, ...},
                "processing_time": {
                    "total_seconds": float,
                    "texts_per_second": float,
                    "formatted": str,
                },
            },
            "individual_results": [  # optional
                {"text": "…", "analysis": {...}},
            ],
            "batch_count": int,  # optional
        }
    """

    BAR_CHAR = "█"

    def __init__(
        self,
        results: Dict[str, Any],
        *,
        compact: bool = False,
        show_confidence: bool = True,
        color: bool = True,
        max_width: Optional[int] = None,
    ) -> None:
        self.results = results
        self.compact = compact
        self.show_confidence = show_confidence
        self.use_color = color and COLOR_ENABLED

        # Try to detect the usable terminal width so visual bars look nice.
        if max_width is None:
            try:
                self.max_width = shutil.get_terminal_size().columns
            except (OSError, AttributeError):  # Fall back if detection fails
                self.max_width = 80
        else:
            self.max_width = max_width

        self.stats = results.get("statistics", {})
        self.total_texts: int = results.get("total_texts", 0)
        self.source: str = results.get("source", "Unknown")
        self.timestamp: str = results.get(
            "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # Short-cuts
        self.sentiment_dist: Dict[str, int] = self.stats.get(
            "sentiment_distribution", {}
        )
        self.emotion_dist: Dict[str, int] = self.stats.get("emotion_distribution", {})
        self.confidence_stats: Dict[str, float] = self.stats.get("confidence_stats", {})
        self.processing_time: Dict[str, Any] = self.stats.get("processing_time", {})

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def generate_summary(self) -> str:
        """Return a formatted text summary."""
        return (
            self._generate_compact_summary()
            if self.compact
            else self._generate_full_summary()
        )

    def export_results(self, fmt: str, file_path: Optional[str] = None) -> str:
        """Write *results* to *file_path* in *fmt* (``csv`` or ``json``)."""
        fmt = fmt.lower()
        if fmt not in {"json", "csv"}:
            raise ValueError(f"Unsupported export format: {fmt}")

        if file_path is None:
            base = os.path.basename(self.source) or "batch_results"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"{base}_{timestamp}.{fmt}"

        if fmt == "json":
            return self._export_json(file_path)
        return self._export_csv(file_path)

    # ------------------------------------------------------------------
    # Compact summary
    # ------------------------------------------------------------------

    def _generate_compact_summary(self) -> str:  # noqa: C901 – readable length
        sep = "=" * self.max_width
        lines = [
            sep,
            self._colored(f"BATCH SUMMARY: {os.path.basename(self.source)}", Fore.CYAN + Style.BRIGHT),
            f"Processed: {self.total_texts} texts | {self.timestamp}",
            sep,
        ]

        # Sentiments inline --------------------------------------------------
        sentiment_parts: list[str] = []
        for sent, count in sorted(self.sentiment_dist.items()):
            if sent == "error" or count == 0:
                continue
            percent = (count / self.total_texts * 100) if self.total_texts else 0
            sentiment_parts.append(
                self._colored(f"{sent}: {percent:.1f}%", self._get_sentiment_color(sent))
            )
        if sentiment_parts:
            lines.append("SENTIMENT: " + " | ".join(sentiment_parts))

        # Emotions inline -----------------------------------------------------
        top_emotions = sorted(self.emotion_dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if top_emotions:
            emo_parts = [
                f"{emo}: {(cnt / self.total_texts * 100):.1f}%" if self.total_texts else f"{emo}: 0%"
                for emo, cnt in top_emotions
            ]
            lines.append("TOP EMOTIONS: " + " | ".join(emo_parts))

        # Confidence / time ---------------------------------------------------
        if self.show_confidence and self.confidence_stats:
            lines.append(f"AVG CONFIDENCE: {self.confidence_stats.get('avg_confidence', 0):.2f}")
        if self.processing_time:
            secs = self.processing_time.get("total_seconds", 0.0)
            rate = self.processing_time.get("texts_per_second", 0.0)
            lines.append(f"PROCESSING: {secs:.2f}s ({rate:.2f} texts/sec)")

        err_rate = self.stats.get("error_rate", 0.0) * 100
        if err_rate:
            lines.append(self._colored(f"ERRORS: {err_rate:.1f}%", Fore.RED))

        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full summary with visual bars
    # ------------------------------------------------------------------

    def _generate_full_summary(self) -> str:  # noqa: C901 – readable length
        sep = "=" * self.max_width
        lines: list[str] = [
            sep,
            self._colored("BATCH PROCESSING SUMMARY", Fore.CYAN + Style.BRIGHT),
            sep,
            f"Source: {self.source}",
            f"Total texts processed: {self.total_texts}",
            f"Timestamp: {self.timestamp}",
        ]

        # Sentiment distribution --------------------------------------------
        lines.append("\n--- Sentiment Distribution ---")
        self._append_bar_section(lines, self.sentiment_dist, color_fn=self._get_sentiment_color)

        # Emotion distribution ----------------------------------------------
        lines.append("\n--- Emotion Distribution ---")
        self._append_bar_section(lines, self.emotion_dist)

        # Confidence statistics ---------------------------------------------
        if self.show_confidence and self.confidence_stats:
            lines.append("\n--- Confidence Statistics ---")
            avg = self.confidence_stats.get("avg_confidence", 0.0)
            mn = self.confidence_stats.get("min_confidence", 0.0)
            mx = self.confidence_stats.get("max_confidence", 0.0)
            std = self.confidence_stats.get("std_dev", 0.0)
            lines.extend([
                f"Average confidence: {avg:.2f}",
                f"Range: {mn:.2f} – {mx:.2f}",
                f"Standard deviation: {std:.2f}" if std else None,
            ])

        # Processing performance --------------------------------------------
        if self.processing_time:
            lines.append("\n--- Processing Performance ---")
            total_s = self.processing_time.get("total_seconds", 0.0)
            rate = self.processing_time.get("texts_per_second", 0.0)
            formatted = self.processing_time.get("formatted", f"{total_s:.2f}s")
            lines.extend([
                f"Total processing time: {formatted}",
                f"Processing rate: {rate:.2f} texts/sec",
                f"Average time per text: {(total_s / self.total_texts * 1000):.2f} ms"
                if self.total_texts
                else None,
            ])
            if "batch_count" in self.results:
                batches = self.results["batch_count"]
                lines.append(f"Number of batches: {batches}")
                if batches:
                    lines.append(f"Average time per batch: {total_s / batches:.2f} s")

        # Error info ---------------------------------------------------------
        err_rate = self.stats.get("error_rate", 0.0) * 100
        if err_rate:
            lines.append("\n--- Error Information ---")
            err_cnt = int(err_rate * self.total_texts / 100) if self.total_texts else 0
            lines.append(self._colored(f"Error rate: {err_rate:.1f}% ({err_cnt} texts)", Fore.RED))
            categories: Dict[str, int] = self.stats.get("error_categories", {})
            if categories:
                lines.append("Error categories:")
                for cat, cnt in sorted(categories.items(), key=lambda kv: kv[1], reverse=True):
                    pct = (cnt / err_cnt * 100) if err_cnt else 0
                    lines.append(f"  - {cat}: {pct:.1f}% ({cnt})")

        lines.append(sep)
        # Remove any *None* placeholders in list comprehension above
        return "\n".join(l for l in lines if l is not None)

    # ------------------------------------------------------------------
    # Bar-drawing helpers
    # ------------------------------------------------------------------

    def _append_bar_section(
        self,
        target: list[str],
        counts: Dict[str, int],
        *,
        color_fn=None,
    ) -> None:
        if not counts:
            target.append("(no data)")
            return
        max_label = max(len(lbl) for lbl in counts)
        bar_width = self.max_width - max_label - 15  # leave room for numbers
        for label, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            if cnt == 0:
                continue
            pct = cnt / self.total_texts * 100 if self.total_texts else 0
            bar_len = int(bar_width * pct / 100)
            bar = self.BAR_CHAR * bar_len
            if color_fn:
                bar = self._colored(bar, color_fn(label))
            target.append(
                f"{label.rjust(max_label)} [{bar.ljust(bar_width)}] {pct:4.1f}% ({cnt})"
            )

    # ------------------------------------------------------------------
    # Colour helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sentiment_color(sentiment: str) -> str:  # noqa: D401
        mapping = {
            "positive": Fore.GREEN,
            "neutral": Fore.BLUE,
            "negative": Fore.RED,
            "error": Fore.YELLOW,
        }
        return mapping.get(sentiment.lower(), "")

    def _colored(self, text: str, colour_code: str) -> str:  # noqa: D401
        if self.use_color and colour_code:
            return f"{colour_code}{text}{Style.RESET_ALL}"
        return text

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.results, fp, indent=2)
        return path

    def _export_csv(self, path: str) -> str:  # noqa: C901 – some complexity unavoidable
        ind_results = self.results.get("individual_results")
        if not ind_results:  # summary-only CSV
            with open(path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Source", self.source])
                writer.writerow(["Total Texts", self.total_texts])
                writer.writerow(["Timestamp", self.timestamp])
                for sent, cnt in self.sentiment_dist.items():
                    writer.writerow([f"Sentiment: {sent}", cnt])
                for emo, cnt in self.emotion_dist.items():
                    writer.writerow([f"Emotion: {emo}", cnt])
                for key, val in self.confidence_stats.items():
                    writer.writerow([f"Confidence: {key}", val])
                for key, val in self.processing_time.items():
                    writer.writerow([f"Processing: {key}", val])
                writer.writerow(["Error Rate", self.stats.get("error_rate", 0)])
            return path

        # Build header from sample results to capture extra fields -------------
        base_cols = [
            "Text",
            "Sentiment",
            "Sentiment_Score",
            "Emotion",
            "Emotion_Score",
            "Error",
        ]
        extra_cols: set[str] = set()
        sample = ind_results[: min(10, len(ind_results))]
        for res in sample:
            if "analysis" not in res or "error" in res:
                continue
            for key, val in res["analysis"].items():
                if key in {"sentiment", "emotion"}:
                    continue
                if isinstance(val, dict):
                    extra_cols.update(f"{key}_{sub}" for sub in val)
                else:
                    extra_cols.add(key)
        header = base_cols + sorted(extra_cols)

        # Write rows -----------------------------------------------------------
        with open(path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            for res in ind_results:
                row = ["" for _ in header]
                row[0] = res.get("text", "")
                if "error" in res:
                    row[header.index("Error")] = res["error"]
                else:
                    analysis = res.get("analysis", {})
                    sent = analysis.get("sentiment", {})
                    emo = analysis.get("emotion", {})
                    row[header.index("Sentiment")] = sent.get("label", "")
                    row[header.index("Sentiment_Score")] = sent.get("score", "")
                    row[header.index("Emotion")] = emo.get("label", "")
                    row[header.index("Emotion_Score")] = emo.get("score", "")
                    for key, val in analysis.items():
                        if key in {"sentiment", "emotion"}:
                            continue
                        if isinstance(val, dict):
                            for sub_key, sub_val in val.items():
                                col_name = f"{key}_{sub_key}"
                                if col_name in header:
                                    row[header.index(col_name)] = sub_val
                        elif key in header:
                            row[header.index(key)] = val
                writer.writerow(row)
        return path


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def generate_batch_report(
    results: Dict[str, Any],
    *,
    compact: bool = False,
    show_confidence: bool = True,
    color: bool = True,
    max_width: Optional[int] = None,
) -> str:
    """Return a formatted report for *results*."""
    return BatchReportGenerator(
        results,
        compact=compact,
        show_confidence=show_confidence,
        color=color,
        max_width=max_width,
    ).generate_summary()


def export_batch_report(
    results: Dict[str, Any],
    fmt: str,
    file_path: Optional[str] = None,
) -> str:
    """Export *results* using :pymeth:`BatchReportGenerator.export_results`."""
    return BatchReportGenerator(results).export_results(fmt, file_path) 