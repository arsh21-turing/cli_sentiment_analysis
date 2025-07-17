import os
import tempfile
import unittest
import csv
import shutil
from pathlib import Path
import random
import string

from src.utils.file_loader import load_data, get_file_iterator, TextFileIterator


class TestFileLoader(unittest.TestCase):
    """Test suite for file_loader.py functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create subdirectories for recursive tests
        
        self.subdir1 = os.path.join(self.test_dir, "subdir1")
        self.subdir2 = os.path.join(self.test_dir, "subdir2")
        self.subdir_nested = os.path.join(self.subdir1, "nested")
        
        os.makedirs(self.subdir1, exist_ok=True)
        os.makedirs(self.subdir2, exist_ok=True)
        os.makedirs(self.subdir_nested, exist_ok=True)
        
        # Sample texts for tests
        self.sample_texts = [
            "This is a test text file with English content.",
            "Here's another sample text for testing purposes.",
            "A third sample with different content.",
            "This is a duplicate of the first line to test duplicate detection.",
            "This is a test text file with English content.",  # Duplicate
            "Short",  # Short text for filtering tests
            "Another duplicate of the first line.",
            "This is a test text file with English content."  # Another duplicate
        ]
        
        # Sample texts with non-ASCII characters (for encoding tests)
        self.utf8_texts = [
            "Text with UTF-8 characters: é è ü ö ñ",
            "More UTF-8: 你好 안녕하세요 привет こんにちは",
            "Mixed UTF-8: €£¥$ and emojis 😊👍🚀"
        ]
        
        # Create test files
        self._create_test_files()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory and all its contents
        shutil.rmtree(self.test_dir)
    
    def _create_test_files(self):
        """Create various test files for the tests."""
        # Create standard text files in the main directory
        for i, text in enumerate(self.sample_texts[:3]):
            with open(os.path.join(self.test_dir, f"file{i+1}.txt"), 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Create some markdown files in the main directory
        for i in range(2):
            with open(os.path.join(self.test_dir, f"markdown{i+1}.md"), 'w', encoding='utf-8') as f:
                f.write(f"# Heading\n\n{self.sample_texts[i]}\n\n## Subheading\n\nMore text here.")
        
        # Create text files in subdirectories
        with open(os.path.join(self.subdir1, "subfile1.txt"), 'w', encoding='utf-8') as f:
            f.write(self.sample_texts[3])
        
        with open(os.path.join(self.subdir2, "subfile2.txt"), 'w', encoding='utf-8') as f:
            f.write(self.sample_texts[4])
        
        with open(os.path.join(self.subdir_nested, "nestedfile.txt"), 'w', encoding='utf-8') as f:
            f.write(self.sample_texts[5])
        
        # Create files with different encodings
        with open(os.path.join(self.test_dir, "utf8.txt"), 'w', encoding='utf-8') as f:
            f.write(self.utf8_texts[0])
        
        with open(os.path.join(self.test_dir, "latin1.txt"), 'w', encoding='latin-1') as f:
            f.write("Latin-1 text with accents: é è ü ö")
        
        # Create a large test file for memory efficiency testing
        self._create_large_file()
        
        # Create CSV files
        self._create_csv_files()
    
    def _create_large_file(self):
        """Create a large text file for testing memory efficiency."""
        large_file_path = os.path.join(self.test_dir, "large_file.txt")
        
        # Generate 1000 random sentences
        sentences = []
        for _ in range(1000):
            sentence_length = random.randint(5, 20)  # Random number of words
            words = [random.choice(string.ascii_lowercase) * random.randint(2, 10) for _ in range(sentence_length)]
            sentence = ' '.join(words) + '.'
            sentences.append(sentence)
        
        # Add some known duplicates
        sentences.extend(sentences[:10])  # Add first 10 sentences again
        random.shuffle(sentences)  # Shuffle to distribute duplicates
        
        # Write the large file
        with open(large_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sentences))
    
    def _create_csv_files(self):
        """Create various CSV files for testing."""
        # Standard CSV with comma separator
        with open(os.path.join(self.test_dir, "standard.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["ID", "Text", "Label"])
            # Rows
            for i, text in enumerate(self.sample_texts[:5]):
                writer.writerow([i+1, text, "label"+str(i+1)])
        
        # CSV without header
        with open(os.path.join(self.test_dir, "noheader.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i, text in enumerate(self.sample_texts[:3]):
                writer.writerow([i+1, text, "label"+str(i+1)])
        
        # CSV with mixed encodings
        with open(os.path.join(self.test_dir, "mixed_encodings.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Text", "Label"])
            for i, text in enumerate(self.utf8_texts):
                writer.writerow([i+1, text, "utf8"])
    
    # Basic functionality tests
    
    def test_load_data_basic(self):
        """Test the basic functionality of load_data with default parameters."""
        texts = load_data(self.test_dir)
        self.assertEqual(len(texts), 6)  # 6 .txt files in the main directory
        self.assertIn(self.sample_texts[0], texts)
        self.assertIn(self.sample_texts[1], texts)
        self.assertIn(self.sample_texts[2], texts)
    
    def test_load_data_specific_file(self):
        """Test loading a specific text file."""
        filepath = os.path.join(self.test_dir, "file1.txt")
        texts = load_data(filepath)
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0], self.sample_texts[0])
    
    def test_load_csv_file(self):
        """Test loading a CSV file."""
        filepath = os.path.join(self.test_dir, "standard.csv")
        texts = load_data(filepath)
        # Should join all columns by default
        self.assertEqual(len(texts), 5)  # 5 data rows
    
    def test_load_csv_specific_column(self):
        """Test loading a specific column from a CSV file."""
        filepath = os.path.join(self.test_dir, "standard.csv")
        texts = load_data(filepath, csv_column=1)  # Text column
        self.assertEqual(len(texts), 5)
        
        # Verify we got the text column
        for i, text in enumerate(self.sample_texts[:5]):
            self.assertIn(text, texts)
    
    # Enhanced feature tests
    
    def test_multiple_extensions(self):
        """Test loading files with multiple extensions."""
        texts = load_data(self.test_dir, extensions=['.txt', '.md'])
        self.assertEqual(len(texts), 8)  # 6 .txt + 2 .md
    
    def test_recursive_loading(self):
        """Test recursive loading of files."""
        texts = load_data(self.test_dir, recursive=True)
        self.assertEqual(len(texts), 9)  # 6 .txt in main + 3 .txt in subdirs
    
    def test_min_length_filtering(self):
        """Test filtering by minimum length."""
        # Set min_length to filter out the short text
        texts = load_data(self.test_dir, recursive=True, min_length=10)
        self.assertEqual(len(texts), 8)  # Should exclude the "Short" text
    
    def test_duplicate_filtering(self):
        """Test filtering out duplicate texts."""
        # Create a directory with duplicates
        dup_dir = os.path.join(self.test_dir, "duplicates")
        os.makedirs(dup_dir, exist_ok=True)
        
        # Create files with duplicate content
        for i, text in enumerate(self.sample_texts):
            with open(os.path.join(dup_dir, f"dup{i+1}.txt"), 'w', encoding='utf-8') as f:
                f.write(text)
        
        # Load with duplicate filtering
        texts = load_data(dup_dir, skip_duplicates=True)
        self.assertEqual(len(texts), 6)  # 8 files but only 6 unique texts
    
    def test_encoding_detection(self):
        """Test automatic encoding detection."""
        # Load files with different encodings
        texts = load_data(self.test_dir, extensions=['.txt'], recursive=False)
        
        # Check that UTF-8 file was loaded correctly
        utf8_content = self.utf8_texts[0]
        self.assertTrue(any(utf8_content in text for text in texts))
        
        # Check that Latin-1 file was loaded
        latin1_text = "Latin-1 text with accents: é è ü ö"
        self.assertTrue(any(latin1_text in text for text in texts))
    
    # Iterator tests
    
    def test_file_iterator_basic(self):
        """Test basic functionality of the file iterator."""
        iterator = get_file_iterator(self.test_dir, batch_size=2)
        # Consume all batches
        batch_count = 0
        for batch in iterator:
            batch_count += 1
            self.assertLessEqual(len(batch), 2)
        self.assertEqual(batch_count, 3)  # 6 files, batch size 2 => 3 batches
        # After iterator is exhausted, next() should raise StopIteration
        with self.assertRaises(StopIteration):
            next(iterator)
        iterator.close()
    
    def test_iterator_with_all_options(self):
        """Test iterator with all options combined."""
        iterator = get_file_iterator(
            file_path=self.test_dir,
            batch_size=3,
            extensions=['.txt', '.md'],
            recursive=True,
            min_length=10,
            skip_duplicates=True
        )
        
        all_texts = []
        for batch in iterator:
            all_texts.extend(batch)
        
        # Should get unique texts of sufficient length
        self.assertGreater(len(all_texts), 0)
        
        # Should not contain short text
        self.assertNotIn("Short", all_texts)
        
        # Should not have duplicates
        self.assertEqual(len(all_texts), len(set(all_texts)))
        
        iterator.close()
    
    def test_csv_iterator(self):
        """Test iterator with CSV files."""
        csv_path = os.path.join(self.test_dir, "standard.csv")
        iterator = get_file_iterator(
            file_path=csv_path,
            batch_size=2,
            csv_column=1  # Text column
        )
        
        all_texts = []
        for batch in iterator:
            all_texts.extend(batch)
        
        # Should get all rows from the text column
        self.assertEqual(len(all_texts), 5)
        for text in self.sample_texts[:5]:
            self.assertIn(text, all_texts)
        
        iterator.close()
    
    def test_csv_no_header(self):
        """Test CSV loading without a header row."""
        csv_path = os.path.join(self.test_dir, "noheader.csv")
        # Use get_file_iterator to specify csv_has_header=False
        iterator = get_file_iterator(csv_path, csv_column=1, csv_has_header=False)
        texts = []
        for batch in iterator:
            texts.extend(batch)
        iterator.close()
        # Should get all rows including the first
        self.assertEqual(len(texts), 3)
    
    # Edge case tests
    
    def test_large_file_memory_efficiency(self):
        """
        Test memory efficiency with large files.
        This test processes a large file in small batches to verify
        the iterator doesn't load everything into memory.
        """
        large_file = os.path.join(self.test_dir, "large_file.txt")
        iterator = get_file_iterator(
            file_path=large_file,
            batch_size=50  # Process 50 lines at a time
        )
        batch_count = 0
        total_lines = 0
        for batch in iterator:
            batch_count += 1
            for text in batch:
                total_lines += len(text.splitlines())
            self.assertLessEqual(len(batch), 50)
        self.assertGreater(batch_count, 0)
        self.assertGreater(total_lines, 1000)
        iterator.close()
    
    def test_mixed_encodings_in_directory(self):
        """Test handling mixed encodings in a directory."""
        # All text files with different encodings
        texts = load_data(
            self.test_dir, 
            extensions=['.txt'],
            recursive=False
        )
        
        # Check that both UTF-8 and Latin-1 were loaded
        self.assertIn(self.utf8_texts[0], texts)
        self.assertIn("Latin-1 text with accents: é è ü ö", texts)
    
    def test_nonexistent_path(self):
        """Test error handling for a nonexistent path."""
        with self.assertRaises(ValueError):
            load_data(os.path.join(self.test_dir, "does_not_exist"))
    
    def test_invalid_file_type(self):
        """Test error handling for invalid file types."""
        # Create a binary file
        binary_path = os.path.join(self.test_dir, "binary.bin")
        with open(binary_path, 'wb') as f:
            f.write(b'\x00\x01\x02\x03')
        
        # Should detect it's not a supported file type
        with self.assertRaises(ValueError):
            load_data(binary_path)
    
    def test_encoding_fallback(self):
        """Test encoding fallback mechanism."""
        # Create a file with windows-1252 encoding for testing fallbacks
        fallback_path = os.path.join(self.test_dir, "windows1252.txt")
        # Only use characters supported by cp1252
        content = "Windows-1252 encoding with special chars: € ™"
        with open(fallback_path, 'w', encoding='windows-1252') as f:
            f.write(content)
        # Try to load it - should detect or fall back to the correct encoding
        texts = load_data(fallback_path)
        self.assertEqual(len(texts), 1)
        self.assertIn("Windows-1252 encoding", texts[0])
    
    def test_duplicate_detection_hash_uniqueness(self):
        """Test the uniqueness of text hashing for duplicate detection."""
        # Create a TextFileIterator instance for direct testing
        iterator = TextFileIterator(self.test_dir, skip_duplicates=True)
        
        # Test different texts generate different hashes
        hash1 = iterator._get_text_hash(self.sample_texts[0])
        hash2 = iterator._get_text_hash(self.sample_texts[1])
        self.assertNotEqual(hash1, hash2)
        
        # Test identical texts generate identical hashes
        hash3 = iterator._get_text_hash(self.sample_texts[0])
        self.assertEqual(hash1, hash3)
        
        iterator.close()
    
    def test_iterator_context_manager(self):
        """Test iterator as a context manager."""
        # Create a text file iterator
        iterator = get_file_iterator(self.test_dir)
        
        # Manually call close to ensure it doesn't raise
        iterator.close()
        
        # Call close again to verify it's idempotent
        iterator.close()  # Should not raise any exceptions
    
    def test_csv_malformed(self):
        """Test handling of malformed CSV files."""
        # Create a malformed CSV file
        malformed_path = os.path.join(self.test_dir, "malformed.csv")
        with open(malformed_path, 'w') as f:
            f.write("This,is,a,header\n")
            f.write("This,is,data\n")  # Missing one field
            f.write("This,is,too,much,data\n")  # Extra field
        
        # Should not raise, but handle the errors gracefully
        texts = load_data(malformed_path)
        self.assertGreaterEqual(len(texts), 2)  # At least header and valid rows


if __name__ == "__main__":
    unittest.main() 