import unittest
import tempfile
import subprocess
import sys
import os
import json
import re
from pathlib import Path

# Local helper import – used only for direct function test
from src.utils.cli import format_result_as_json


class TestCliJsonStream(unittest.TestCase):
    """Verify `--json-stream` emits NDJSON without ANSI codes and with expected keys."""

    def setUp(self):
        # Two-line batch file with different sentiments
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt")
        tmp.write("I love this product! It's amazing.\n")
        tmp.write("I'm disappointed with the quality; it broke immediately.\n")
        tmp.close()
        self.batch_path = tmp.name

        # Regex to detect ANSI escapes
        self.ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        # Resolve CLI entry script
        self.main_script = Path(__file__).resolve().parent.parent / "src" / "main.py"
        if not self.main_script.exists():
            self.skipTest("main.py not found – CLI not available")

    def tearDown(self):
        try:
            os.unlink(self.batch_path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    def test_json_stream_subprocess(self):
        """Run CLI with `--json-stream` and ensure two clean JSON lines are produced."""
        cmd = [
            sys.executable,
            str(self.main_script),
            "--file",
            self.batch_path,
            "--json-stream",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        lines = [ln for ln in result.stdout.split("\n") if ln.strip()]
        self.assertEqual(len(lines), 2, f"Expected 2 JSON lines, got {len(lines)}")

        for line in lines:
            # No ANSI codes
            self.assertIsNone(self.ansi.search(line), "ANSI codes found in JSON output")
            # Must be valid JSON with required keys
            obj = json.loads(line)
            for key in ("text", "sentiment", "emotion"):
                self.assertIn(key, obj)

    # ------------------------------------------------------------------
    def test_format_result_as_json_helper(self):
        """Directly exercise `format_result_as_json` utility."""
        sample = {
            "text": "Test msg",
            "sentiment": {"label": "positive", "score": 0.9},
            "emotion": {"label": "joy", "score": 0.8},
        }
        js = format_result_as_json(sample)
        self.assertIsNone(self.ansi.search(js))
        parsed = json.loads(js)
        self.assertEqual(parsed["text"], sample["text"]) 