import os
import tempfile
import unittest
import csv
import shutil
from unittest.mock import Mock, patch
from datetime import datetime

from src.utils.csv_chunker import CSVChunker, process_large_csv


class TestCSVChunker(unittest.TestCase):
    """Test suite for CSV chunking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Sample data
        self.headers = ["ID", "Text", "Category", "Score"]
        self.data = [
            [1, "Sample text one", "A", 0.85],
            [2, "Sample text two", "B", 0.72],
            [3, "Sample text three", "A", 0.91],
            [4, "Sample text four", "C", 0.64],
            [5, "Short", "D", 0.45]
        ]
        
        # Create test files
        self._create_test_files()
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.test_dir)
    
    def _create_test_files(self):
        """Create test CSV files."""
        # Standard CSV
        with open(os.path.join(self.test_dir, "standard.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.data)
        
        # CSV with semicolon delimiter
        with open(os.path.join(self.test_dir, "semicolon.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(self.headers)
            writer.writerows(self.data)
        
        # CSV without header
        with open(os.path.join(self.test_dir, "noheader.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(self.data)
        
        # CSV with duplicates
        with open(os.path.join(self.test_dir, "duplicates.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.data)
            writer.writerows(self.data[:2])  # Add some duplicates
    
    def test_basic_functionality(self):
        """Test basic CSV chunking."""
        chunker = CSVChunker(os.path.join(self.test_dir, "standard.csv"))
        chunks = list(chunker)
        
        # Should get all texts
        all_texts = [item for chunk in chunks for item in chunk]
        self.assertEqual(len(all_texts), len(self.data))
        
        # Verify content
        self.assertTrue(any("Sample text one" in text for text in all_texts))
    
    def test_chunk_size(self):
        """Test custom chunk sizes."""
        chunker = CSVChunker(
            os.path.join(self.test_dir, "standard.csv"),
            chunk_size=2
        )
        chunks = list(chunker)
        
        # Should have correct number of chunks
        expected_chunks = (len(self.data) + 1) // 2
        self.assertEqual(len(chunks), expected_chunks)
        
        # Each chunk should have correct size
        for chunk in chunks[:-1]:
            self.assertEqual(len(chunk), 2)
    
    def test_delimiter_detection(self):
        """Test delimiter auto-detection."""
        chunker = CSVChunker(os.path.join(self.test_dir, "semicolon.csv"))
        self.assertEqual(chunker.delimiter, ';')
        
        # Verify we can read the data
        chunks = list(chunker)
        all_texts = [item for chunk in chunks for item in chunk]
        self.assertEqual(len(all_texts), len(self.data))
    
    def test_header_detection(self):
        """Test header detection."""
        # With header
        chunker1 = CSVChunker(os.path.join(self.test_dir, "standard.csv"))
        self.assertTrue(chunker1.has_header)
        
        # Without header
        chunker2 = CSVChunker(os.path.join(self.test_dir, "noheader.csv"))
        self.assertFalse(chunker2.has_header)
    
    def test_text_column_selection(self):
        """Test text column selection."""
        # Use Category column instead of Text
        chunker = CSVChunker(
            os.path.join(self.test_dir, "standard.csv"),
            text_columns=2
        )
        chunks = list(chunker)
        all_texts = [item for chunk in chunks for item in chunk]
        
        # Should contain categories
        self.assertTrue(any("A" in text for text in all_texts))
        self.assertTrue(any("B" in text for text in all_texts))
    
    def test_minimum_length_filter(self):
        """Test minimum length filtering."""
        chunker = CSVChunker(
            os.path.join(self.test_dir, "standard.csv"),
            min_length=10
        )
        chunks = list(chunker)
        all_texts = [item for chunk in chunks for item in chunk]
        
        # Should exclude "Short" text
        self.assertFalse(any("Short" == text for text in all_texts))
        self.assertEqual(len(all_texts), len(self.data) - 1)
    
    def test_duplicate_filtering(self):
        """Test duplicate text filtering."""
        chunker = CSVChunker(
            os.path.join(self.test_dir, "duplicates.csv"),
            skip_duplicates=True
        )
        chunks = list(chunker)
        all_texts = [item for chunk in chunks for item in chunk]
        
        # Should only have unique texts
        self.assertEqual(len(all_texts), len(self.data))
    
    @patch('src.utils.batch._process_sequential')
    def test_process_large_csv(self, mock_process):
        """Test process_large_csv function."""
        # Mock analyzer and processing
        mock_analyzer = Mock()
        mock_process.return_value = [
            {"text": text, "analysis": {"result": "mock"}}
            for text in ["Sample text one", "Sample text two"]
        ]
        
        # Process CSV
        results = process_large_csv(
            file_path=os.path.join(self.test_dir, "standard.csv"),
            analyzer=mock_analyzer,
            batch_size=2
        )
        
        # Verify results structure
        self.assertIn("source", results)
        self.assertIn("total_texts", results)
        self.assertIn("batch_count", results)
        self.assertIn("timestamp", results)
        self.assertIn("individual_results", results)
        self.assertIn("statistics", results)
    
    def test_progress_tracking(self):
        """Test progress tracking."""
        chunker = CSVChunker(
            os.path.join(self.test_dir, "standard.csv"),
            chunk_size=2
        )
        
        # Get first chunk
        next(iter(chunker))
        
        # Check progress
        processed, total, progress = chunker.get_progress()
        self.assertGreater(processed, 0)
        self.assertEqual(total, os.path.getsize(os.path.join(self.test_dir, "standard.csv")))
        self.assertGreater(progress, 0)
        self.assertLess(progress, 100)


if __name__ == '__main__':
    unittest.main() 