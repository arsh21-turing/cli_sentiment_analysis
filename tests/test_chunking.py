import unittest
from unittest.mock import Mock, patch
import sys
import os
import random

# Add the project root to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.chunking import split_text, process_chunked_text, _combine_chunk_results

class TestChunking(unittest.TestCase):
    def test_split_text_empty(self):
        """Test splitting empty text."""
        result = split_text("")
        self.assertEqual(result, [])
    
    def test_split_text_small(self):
        """Test splitting text smaller than max chunk size."""
        text = "This is a small text that should fit in one chunk."
        result = split_text(text, max_chunk_size=100)
        self.assertEqual(result, [text])
    
    def test_split_text_exact_size(self):
        """Test splitting text that exactly matches max chunk size."""
        text = "A" * 100
        result = split_text(text, max_chunk_size=100)
        self.assertEqual(result, [text])
    
    def test_split_text_sentence_boundaries(self):
        """Test that text is split at sentence boundaries when possible."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        result = split_text(text, max_chunk_size=30)
        self.assertEqual(len(result), 3)
        self.assertTrue(all("." in chunk for chunk in result[:2]))
        
        # Verify each sentence is in its own chunk
        self.assertEqual(result[0], "This is sentence one.")
        self.assertEqual(result[1], " This is sentence two.")
        self.assertEqual(result[2], " This is sentence three.")
    
    def test_split_text_long_sentence(self):
        """Test handling of sentences longer than max chunk size."""
        long_sentence = "This is an extremely long sentence that should be split into multiple chunks because it exceeds the maximum chunk size limit that we have established for this test case."
        result = split_text(long_sentence, max_chunk_size=50)
        self.assertTrue(len(result) > 1)
        # Total characters should match original (minus any trailing spaces that might be trimmed)
        self.assertEqual(sum(len(chunk) for chunk in result), len(long_sentence))
    
    def test_split_text_mixed_lengths(self):
        """Test text with a mix of short and long sentences."""
        text = "Short. Another short sentence. " + "X" * 100 + ". Short again."
        result = split_text(text, max_chunk_size=50)
        self.assertTrue(len(result) >= 3)  # At least 3 chunks
        
        # First chunk should contain the short sentences
        self.assertTrue(result[0].startswith("Short. Another short sentence."))
        
        # Last chunk should contain "Short again."
        self.assertTrue(any("Short again." in chunk for chunk in result))
    
    def test_split_text_with_overlap(self):
        """Test splitting text with overlap between chunks."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = split_text(text, max_chunk_size=25, overlap=10)
        self.assertTrue(len(result) > 1)
        
        # Check that chunks overlap
        for i in range(len(result) - 1):
            chunk = result[i]
            next_chunk = result[i + 1]
            # The end of one chunk should be at the start of the next
            overlap_text = chunk[-10:] if len(chunk) >= 10 else chunk
            self.assertTrue(next_chunk.startswith(overlap_text) or 
                          next_chunk[:10].strip() in chunk)
    
    def test_split_text_large_document(self):
        """Test splitting a large document with many sentences."""
        # Generate a large document with varied sentence lengths
        sentences = []
        for _ in range(100):
            length = random.randint(5, 50)
            sentences.append("X" * length + ".")
        large_text = " ".join(sentences)
        
        # Split with a small chunk size
        result = split_text(large_text, max_chunk_size=200)
        
        # Verify we have multiple chunks
        self.assertTrue(len(result) > 10)
        
        # Verify the total content matches (accounting for spaces that might be added or removed)
        original_no_spaces = large_text.replace(" ", "")
        result_no_spaces = "".join(result).replace(" ", "")
        self.assertEqual(len(original_no_spaces), len(result_no_spaces))

    def test_process_chunked_text_small_text(self):
        """Test processing small text that doesn't need chunking."""
        # Create a mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = {
            "sentiment": {"label": "positive", "score": 0.8},
            "emotion": {"label": "joy", "score": 0.7}
        }
        
        text = "This is a small text."
        result = process_chunked_text(text, mock_analyzer, max_chunk_size=1000)
        
        # Should call analyze only once and not chunk
        mock_analyzer.analyze.assert_called_once_with(text)
        self.assertFalse(result.get("chunked", False))
    
    def test_combine_chunk_results_all_errors(self):
        """Test combining chunk results when all chunks have errors."""
        chunk_results = [
            {"chunk_index": 0, "chunk_size": 100, "error": "Error 1"},
            {"chunk_index": 1, "chunk_size": 150, "error": "Error 2"}
        ]
        
        result = _combine_chunk_results(chunk_results)
        self.assertEqual(result["error"], "All chunks failed analysis")
        self.assertTrue(result["chunked"])
    
    def test_combine_chunk_results_partial_errors(self):
        """Test combining results when some chunks have errors."""
        chunk_results = [
            {"chunk_index": 0, "chunk_size": 100, "error": "Error 1"},
            {
                "chunk_index": 1, 
                "chunk_size": 200, 
                "result": {
                    "sentiment": {"label": "positive", "score": 0.8},
                    "emotion": {"label": "joy", "score": 0.7}
                }
            }
        ]
        
        result = _combine_chunk_results(chunk_results)
        
        # Should use results from successful chunks
        self.assertEqual(result["sentiment"]["label"], "positive")
        self.assertEqual(result["sentiment"]["score"], 0.8)
        self.assertEqual(result["emotion"]["label"], "joy")
        self.assertEqual(result["emotion"]["score"], 0.7)

    def test_split_text_complex_punctuation(self):
        """Test splitting text with complex punctuation and quotations."""
        text = 'He said, "This is a quote!" Then he continued. "Another quote?" Yes, it is.'
        result = split_text(text, max_chunk_size=30)
        
        # Verify we get multiple chunks
        self.assertTrue(len(result) > 1)
        
        # Verify we don't split inside quotation marks (if possible)
        for chunk in result:
            # Count quotes in each chunk
            quotes_count = chunk.count('"')
            # Either no quotes or an even number (open and close)
            self.assertTrue(quotes_count == 0 or quotes_count % 2 == 0, 
                         f"Chunk '{chunk}' has unbalanced quotes")

if __name__ == '__main__':
    unittest.main() 