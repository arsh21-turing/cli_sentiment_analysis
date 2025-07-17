import os
import tempfile
import unittest
import shutil
from unittest.mock import Mock, patch

from src.utils.batch import process_batch, process_with_iterator, export_batch_results, display_batch_summary
from src.utils.file_loader import get_file_iterator


class MockAnalyzer:
    """Mock analyzer for testing batch processing."""
    
    def analyze(self, text):
        """Mock analysis function."""
        return {
            "sentiment": {
                "label": "positive" if "good" in text.lower() else "negative" if "bad" in text.lower() else "neutral",
                "score": 0.8 if "good" in text.lower() else 0.2 if "bad" in text.lower() else 0.5
            },
            "emotion": {
                "label": "joy" if "good" in text.lower() else "anger" if "bad" in text.lower() else "neutral",
                "score": 0.75 if "good" in text.lower() else 0.65 if "bad" in text.lower() else 0.5
            }
        }


class TestBatch(unittest.TestCase):
    """Test suite for batch.py functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.texts = [
            "This is a good test text.",
            "This is a bad example.",
            "Neutral text without sentiment.",
            "Another good positive message.",
            "Something terrible and bad happened."
        ]
        
        # Create test text files
        for i, text in enumerate(self.texts):
            with open(os.path.join(self.test_dir, f"test{i+1}.txt"), "w") as f:
                f.write(text)
        
        # Create mock analyzer
        self.analyzer = MockAnalyzer()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and all its contents
        shutil.rmtree(self.test_dir)
    
    def test_process_batch_basic(self):
        """Test basic batch processing functionality."""
        results = process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer,
            batch_size=2  # Process in batches of 2
        )
        
        # Check basic structure of results
        self.assertEqual(results["total_texts"], 5)
        self.assertEqual(results["batch_count"], 3)  # 2+2+1
        self.assertEqual(len(results["individual_results"]), 5)
        
        # Check statistics
        stats = results["statistics"]
        self.assertEqual(stats["sentiment_distribution"]["positive"], 2)  # 2 positive texts
        self.assertEqual(stats["sentiment_distribution"]["negative"], 2)  # 2 negative texts
        self.assertEqual(stats["sentiment_distribution"]["neutral"], 1)   # 1 neutral text
        
        self.assertEqual(stats["emotion_distribution"]["joy"], 2)        # 2 joy texts
        self.assertEqual(stats["emotion_distribution"]["anger"], 2)       # 2 anger texts
        self.assertEqual(stats["emotion_distribution"]["neutral"], 1)     # 1 neutral text
    
    def test_process_with_iterator(self):
        """Test processing with an existing iterator."""
        # Create iterator
        iterator = get_file_iterator(
            file_path=self.test_dir,
            batch_size=3
        )
        
        # Process with iterator
        results = process_with_iterator(
            file_iterator=iterator,
            analyzer=self.analyzer
        )
        
        # Check results
        self.assertEqual(results["total_texts"], 5)
        self.assertEqual(results["batch_count"], 2)  # 3+2
        self.assertEqual(len(results["individual_results"]), 5)
    
    def test_parallel_processing(self):
        """Test parallel processing of texts."""
        # Process in parallel
        results = process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer,
            parallel=True,
            max_workers=2
        )
        
        # Should get same results as sequential
        self.assertEqual(results["total_texts"], 5)
        self.assertEqual(len(results["individual_results"]), 5)
        
        # Verify all texts were analyzed correctly
        sentiments = [r["analysis"]["sentiment"]["label"] for r in results["individual_results"]]
        self.assertEqual(sentiments.count("positive"), 2)
        self.assertEqual(sentiments.count("negative"), 2)
        self.assertEqual(sentiments.count("neutral"), 1)
    
    def test_export_results_json(self):
        """Test exporting results to JSON."""
        # Process batch
        results = process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer
        )
        
        # Export to JSON
        json_path = export_batch_results(results, "json")
        
        # Verify file exists and is readable
        self.assertTrue(os.path.exists(json_path))
        try:
            import json
            with open(json_path, "r") as f:
                exported_data = json.load(f)
                
            # Verify contents
            self.assertEqual(exported_data["total_texts"], 5)
            
            # Clean up the exported file
            os.remove(json_path)
        except Exception as e:
            self.fail(f"Failed to read exported JSON: {e}")
    
    def test_export_results_csv(self):
        """Test exporting results to CSV."""
        # Process batch
        results = process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer
        )
        
        # Export to CSV
        csv_path = export_batch_results(results, "csv")
        
        # Verify file exists and is readable
        self.assertTrue(os.path.exists(csv_path))
        try:
            import csv
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
                
            # Verify header and row count
            self.assertEqual(len(rows), 6)  # Header + 5 data rows
            self.assertEqual(rows[0][0], "Text")
            self.assertEqual(rows[0][1], "Sentiment")
            
            # Clean up the exported file
            os.remove(csv_path)
        except Exception as e:
            self.fail(f"Failed to read exported CSV: {e}")
    
    def test_error_handling(self):
        """Test error handling during batch processing."""
        # Create an analyzer that raises an error for specific text
        def analyze_with_error(text):
            if "bad" in text:
                raise ValueError("Error analyzing negative text")
            return {
                "sentiment": {"label": "positive", "score": 0.8},
                "emotion": {"label": "joy", "score": 0.8}
            }
        
        error_analyzer = Mock()
        error_analyzer.analyze = analyze_with_error
        
        # Process batch with error-raising analyzer
        results = process_batch(
            file_path=self.test_dir,
            analyzer=error_analyzer
        )
        
        # Check error handling
        error_count = 0
        for result in results["individual_results"]:
            if "error" in result:
                error_count += 1
                self.assertIn("Error analyzing negative text", result["error"])
        
        # Should be 2 errors for the "bad" texts
        self.assertEqual(error_count, 2)
        
        # Should be reflected in statistics
        self.assertEqual(results["statistics"]["sentiment_distribution"]["error"], 2)
        self.assertGreater(results["statistics"]["error_rate"], 0)
    
    def test_custom_file_loader_options(self):
        """Test batch processing with custom file loader options."""
        # Create some additional files
        with open(os.path.join(self.test_dir, "duplicate.txt"), "w") as f:
            f.write(self.texts[0])  # Duplicate of first text
            
        with open(os.path.join(self.test_dir, "short.txt"), "w") as f:
            f.write("Short")  # Short text
        
        # Process with custom options
        results = process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer,
            skip_duplicates=True,   # Skip duplicates
            min_length=10           # Skip short texts
        )
        
        # Should skip the duplicates and short texts
        self.assertEqual(results["total_texts"], 5)  # Original 5 texts, not 7
    
    def test_statistics_calculation(self):
        """Test accuracy of statistical calculations."""
        results = process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer
        )
        
        # Verify statistics calculations
        stats = results["statistics"]
        
        # Check sentiment distribution
        self.assertEqual(stats["sentiment_distribution"]["positive"], 2)
        self.assertEqual(stats["sentiment_distribution"]["negative"], 2)
        self.assertEqual(stats["sentiment_distribution"]["neutral"], 1)
        
        # Check confidence stats
        self.assertIn("avg_confidence", stats["confidence_stats"])
        self.assertIn("min_confidence", stats["confidence_stats"])
        self.assertIn("max_confidence", stats["confidence_stats"])
        
        # Verify correct values (based on our mock analyzer)
        self.assertEqual(stats["confidence_stats"]["min_confidence"], 0.2)
        self.assertEqual(stats["confidence_stats"]["max_confidence"], 0.8)
        
        # Average should be (0.8 + 0.2 + 0.5 + 0.8 + 0.2) / 5 = 0.5
        self.assertAlmostEqual(stats["confidence_stats"]["avg_confidence"], 0.5)
    
    def test_processing_time_tracking(self):
        """Test that processing time is tracked correctly."""
        results = process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer
        )
        
        # Verify processing time information is present
        self.assertIn("processing_time", results["statistics"])
        self.assertIn("total_seconds", results["statistics"]["processing_time"])
        self.assertIn("texts_per_second", results["statistics"]["processing_time"])
        self.assertIn("formatted", results["statistics"]["processing_time"])
        
        # Values should be reasonable
        self.assertGreaterEqual(results["statistics"]["processing_time"]["total_seconds"], 0)
        self.assertGreaterEqual(results["statistics"]["processing_time"]["texts_per_second"], 0)
    
    def test_process_empty_directory(self):
        """Test processing an empty directory."""
        # Create empty directory
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            process_batch(
                file_path=empty_dir,
                analyzer=self.analyzer
            )
    
    @patch('tqdm.tqdm')
    def test_progress_bar_usage(self, mock_tqdm):
        """Test that progress bars are used correctly."""
        # Process batch
        process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer
        )
        
        # Verify tqdm was called
        self.assertTrue(mock_tqdm.called)
    
    def test_display_batch_summary(self):
        """Test the display_batch_summary function."""
        # Process batch
        results = process_batch(
            file_path=self.test_dir,
            analyzer=self.analyzer
        )
        
        # Capture stdout to verify output
        from io import StringIO
        import sys
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Display summary
            display_batch_summary(results)
            
            # Get output
            output = captured_output.getvalue()
            
            # Verify key information is present
            self.assertIn("Batch Processing Summary", output)
            self.assertIn("Total texts processed: 5", output)
            self.assertIn("Sentiment Distribution", output)
            self.assertIn("Emotion Distribution", output)
            self.assertIn("Confidence Stats", output)
            self.assertIn("Error Rate", output)
        finally:
            sys.stdout = sys.__stdout__  # Restore stdout


if __name__ == "__main__":
    unittest.main() 