# tests/test_chunking_integration.py
"""
Integration tests for the chunking pipeline.

These tests verify the end‑to‑end flow:
CLI → batch processor → chunking utility → analyzer → final results
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import shutil
from io import StringIO

# Add project root to PYTHONPATH so we can import src.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports from the project
from src.utils.cli import parse_args, main
from src.utils.batch import process_batch
from src.utils.chunking import process_chunked_text, split_text



# Helper functions / fixtures


def _create_sample_files(tmp_dir: str):
    """Create small / medium / large text files plus a CSV."""
    # Small (no chunking expected)
    with open(os.path.join(tmp_dir, "small.txt"), "w") as f:
        f.write("Short file – chunking not required.")

    # Medium (may or may not chunk)
    with open(os.path.join(tmp_dir, "medium.txt"), "w") as f:
        f.write("Medium file sentence. " * 50)          # ~1 000 chars

    # Large (will definitely chunk)
    with open(os.path.join(tmp_dir, "large.txt"), "w") as f:
        for i in range(120):
            sentiment = ["positive", "neutral", "negative"][i % 3]
            f.write(f"{sentiment.capitalize()} sentence {i}. ")

    # Very‑large file to test thresholds / weighting
    with open(os.path.join(tmp_dir, "very_large.txt"), "w") as f:
        sections = [
            "Positive section. " * 120,
            "Neutral section. " * 120,
            "Negative section. " * 120,
        ]
        f.write("\n\n".join(sections))

    # CSV input
    with open(os.path.join(tmp_dir, "sample.csv"), "w") as f:
        f.write("id,text\n")
        f.write("1,Small csv row.\n")
        f.write("2," + ("Row with medium size text. " * 40) + "\n")
        f.write("3," + ("Row with large text. " * 120) + "\n")


def _build_lightweight_analyzer():
    """
    Returns a lightweight rule‑based analyzer mock that mimics
    SentimentEmotionTransformer.analyze(text).
    """
    mock_analyzer = MagicMock()

    def simple_analysis(txt: str):
        txt = txt.lower()
        pos = sum(w in txt for w in ("good", "great", "positive", "happy", "joy"))
        neg = sum(w in txt for w in ("bad", "negative", "terrible", "sad"))
        if pos > neg:
            label, score = "positive", 0.7
        elif neg > pos:
            label, score = "negative", 0.7
        else:
            label, score = "neutral", 0.6

        return {
            "sentiment": {"label": label, "score": score},
            "emotion": {"label": "joy" if label == "positive" else "neutral", "score": 0.6},
        }

    mock_analyzer.analyze.side_effect = simple_analysis
    return mock_analyzer



# Integration‑test class


class TestChunkingIntegration(unittest.TestCase):
    """End‑to‑end integration tests for chunking + batch pipeline."""

    def setUp(self):
        # Temporary workspace
        self.tmp_dir = tempfile.mkdtemp(prefix="chunk_int_")
        _create_sample_files(self.tmp_dir)

        # Lightweight analyzer used in many tests
        self.analyzer = _build_lightweight_analyzer()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    # -------------  process_chunked_text integration ---------------------- #

    def test_process_chunked_text_combines_weights_correctly(self):
        """Weighted average label should match majority of text length."""
        # craft text where majority (70%) is negative
        text = ("positive. " * 30) + ("negative. " * 70)
        res = process_chunked_text(
            text=text,
            analyzer=self.analyzer,
            max_chunk_size=100,
            overlap=0,
        )
        self.assertTrue(res["chunked"])
        self.assertEqual(res["sentiment"]["label"], "negative")



# Run via `python -m unittest tests.test_chunking_integration`


if __name__ == "__main__":
    unittest.main()
