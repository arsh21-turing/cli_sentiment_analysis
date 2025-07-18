import unittest
import tempfile
import os

from src.utils.settings import Settings


class TestSettings(unittest.TestCase):
    """Unit-tests for :class:`Settings`. They focus on the features added recently
    (summary-only, quiet, json-stream modes; threshold setters; persistence)."""

    def setUp(self):
        self.settings = Settings()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        self.temp_path = tmp.name

    def tearDown(self):
        try:
            os.unlink(self.temp_path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    def test_defaults(self):
        self.assertEqual(self.settings.get_sentiment_threshold(), Settings.DEFAULT_SENTIMENT_THRESHOLD)
        self.assertEqual(self.settings.get_emotion_threshold(), Settings.DEFAULT_EMOTION_THRESHOLD)
        self.assertFalse(self.settings.summary_only)
        self.assertFalse(self.settings.is_quiet)
        self.assertFalse(self.settings.json_stream)

    # ------------------------------------------------------------------
    # Mode toggles
    # ------------------------------------------------------------------
    def test_mode_toggles(self):
        self.settings.set_summary_only(True)
        self.assertTrue(self.settings.summary_only)
        self.settings.set_quiet_mode(True)
        self.assertTrue(self.settings.is_quiet)
        self.settings.set_json_stream(True)
        self.assertTrue(self.settings.json_stream)

        # Toggle back
        self.settings.set_summary_only(False)
        self.settings.set_quiet_mode(False)
        self.settings.set_json_stream(False)
        self.assertFalse(self.settings.summary_only)
        self.assertFalse(self.settings.is_quiet)
        self.assertFalse(self.settings.json_stream)

    # ------------------------------------------------------------------
    # Threshold manipulation & validation
    # ------------------------------------------------------------------
    def test_threshold_updates(self):
        # Valid updates
        self.settings.set_sentiment_threshold(0.75)
        self.assertEqual(self.settings.get_sentiment_threshold(), 0.75)
        self.settings.set_emotion_threshold(0.65)
        self.assertEqual(self.settings.get_emotion_threshold(), 0.65)

        # Out-of-range values should be ignored
        original = self.settings.get_sentiment_threshold()
        self.settings.set_sentiment_threshold(1.5)
        self.assertEqual(self.settings.get_sentiment_threshold(), original)

    def test_multi_level_thresholds(self):
        self.settings.set_sentiment_threshold_levels(0.9, 0.7, 0.5)
        levels = self.settings.get_sentiment_threshold_levels()
        self.assertEqual(levels, {"high": 0.9, "medium": 0.7, "low": 0.5})

        # Invalid ordering should not mutate existing levels
        self.settings.set_sentiment_threshold_levels(0.5, 0.7, 0.9)
        self.assertEqual(self.settings.get_sentiment_threshold_levels(), levels)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def test_reset_to_defaults(self):
        self.settings.set_sentiment_threshold(0.9).set_emotion_threshold(0.2)
        self.settings.set_summary_only(True).set_quiet_mode(True).set_json_stream(True)
        self.settings.reset_to_defaults()
        self.assertEqual(self.settings.get_sentiment_threshold(), Settings.DEFAULT_SENTIMENT_THRESHOLD)
        self.assertEqual(self.settings.get_emotion_threshold(), Settings.DEFAULT_EMOTION_THRESHOLD)
        self.assertFalse(self.settings.summary_only)
        self.assertFalse(self.settings.is_quiet)
        self.assertFalse(self.settings.json_stream)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def test_save_and_load(self):
        self.settings.set_sentiment_threshold(0.66).set_emotion_threshold(0.44)
        ok = self.settings.save_settings(self.temp_path)
        self.assertTrue(ok, "save_settings should return True on success")

        fresh = Settings().load_settings(self.temp_path)
        self.assertAlmostEqual(fresh.get_sentiment_threshold(), 0.66)
        self.assertAlmostEqual(fresh.get_emotion_threshold(), 0.44)


if __name__ == "__main__":
    unittest.main(verbosity=2) 