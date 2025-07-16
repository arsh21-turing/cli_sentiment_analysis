import unittest
import os
import sys
import tempfile
import subprocess
import re
from pathlib import Path


class TestCliNoColour(unittest.TestCase):
    """Verify behaviour of the `--no-colour` flag.

    We run the CLI twice – once with colours (default) and once with
    `--no-colour` – then ensure:
      1. Colourful run contains ANSI escape sequences.
      2. No-colour run contains **no** ANSI escapes.
      3. After stripping colour, the textual content is identical.
    Additional checks cover combined flags and direct `--text` usage.
    """

    def setUp(self):
        # Temporary two-line batch file
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt")
        tmp.write("I'm really happy with this product! It exceeded my expectations.\n")
        tmp.write("This is disappointing. The quality is much lower than anticipated.\n")
        tmp.close()
        self.batch_path = tmp.name

        # Regex for ANSI escape codes
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        # Resolve `src/main.py`
        self.project_root = Path(__file__).resolve().parent.parent
        self.main_script = self.project_root / "src" / "main.py"
        if not self.main_script.exists():
            self.skipTest(f"main.py not found at {self.main_script}")

    def tearDown(self):
        try:
            os.unlink(self.batch_path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _strip_ansi(self, text: str) -> str:
        return self.ansi_escape.sub("", text)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_colour_vs_no_colour_batch(self):
        """Run with colours and with `--no-colour`; compare outputs."""
        cmd_colour = [sys.executable, str(self.main_script), "--file", self.batch_path]
        cmd_plain = [sys.executable, str(self.main_script), "--file", self.batch_path, "--no-colour"]

        res_colour = subprocess.run(cmd_colour, capture_output=True, text=True, check=True)
        res_plain = subprocess.run(cmd_plain, capture_output=True, text=True, check=True)

        coloured_out = res_colour.stdout
        plain_out = res_plain.stdout

        # 1. coloured_out must contain ANSI codes
        self.assertIsNotNone(self.ansi_escape.search(coloured_out), "Expected ANSI codes in coloured output")
        # 2. plain_out must NOT contain ANSI codes
        self.assertIsNone(self.ansi_escape.search(plain_out), "Did not expect ANSI codes in --no-colour output")
        # 3. After stripping, text must match (ignoring whitespace diff)
        stripped_colour = " ".join(self._strip_ansi(coloured_out).split())
        stripped_plain = " ".join(plain_out.split())
        self.assertEqual(stripped_colour, stripped_plain, "Text content differs between coloured and plain outputs")

        # Ensure both lines were processed
        self.assertIn("happy", stripped_plain.lower())
        self.assertIn("disappoint", stripped_plain.lower())

    def test_quiet_summary_only_no_colour(self):
        """Combination of `--quiet --summary-only --no-colour` should yield two summary lines, no ANSI."""
        cmd = [
            sys.executable,
            str(self.main_script),
            "--file",
            self.batch_path,
            "--quiet",
            "--summary-only",
            "--no-colour",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = res.stdout

        self.assertIsNone(self.ansi_escape.search(out), "Unexpected ANSI codes in quiet/summary-only output")
        # No progress/info lines expected
        self.assertNotIn("Processing", out)
        # Exactly two non-empty lines – one per input row
        lines = [ln for ln in out.split("\n") if ln.strip()]
        self.assertEqual(len(lines), 2, f"Expected 2 summary lines, got {len(lines)}")
        for line in lines:
            self.assertIn("sentiment", line.lower())
            self.assertIn("emotion", line.lower())

    def test_text_input_no_colour(self):
        text = "I'm feeling really happy today!"
        cmd_colour = [sys.executable, str(self.main_script), "--text", text]
        cmd_plain = [sys.executable, str(self.main_script), "--text", text, "--no-colour"]

        res_colour = subprocess.run(cmd_colour, capture_output=True, text=True, check=True)
        res_plain = subprocess.run(cmd_plain, capture_output=True, text=True, check=True)

        self.assertIsNotNone(self.ansi_escape.search(res_colour.stdout), "Expected ANSI codes in coloured run")
        self.assertIsNone(self.ansi_escape.search(res_plain.stdout), "No ANSI codes expected in --no-colour run")

        stripped_colour = " ".join(self._strip_ansi(res_colour.stdout).split())
        stripped_plain = " ".join(res_plain.stdout.split())
        self.assertEqual(stripped_colour, stripped_plain)


if __name__ == "__main__":
    unittest.main(verbosity=2) 