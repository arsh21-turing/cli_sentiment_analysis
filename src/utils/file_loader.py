import os
import csv
import glob
import hashlib
import chardet
from typing import List, Iterator, Set, Optional, Dict, Union, Tuple, Any


class TextFileIterator:
    """
    Iterator for reading text files in batches.
    
    This class provides an iterator interface for reading text files from 
    a directory (recursively) or CSV files. It supports various text formats,
    handles encoding detection, and can filter duplicates and short lines.
    """
    
    def __init__(self, 
                 path: str, 
                 extensions: List[str] = None,
                 csv_column: int = None,
                 csv_has_header: bool = True,
                 min_length: int = 0,
                 skip_duplicates: bool = False,
                 recursive: bool = False,
                 encoding: str = None,
                 batch_size: int = 100):
        """
        Initialize a text file iterator.
        
        Parameters:
        - path (str): Path to either a directory or a CSV file
        - extensions (List[str], optional): List of file extensions to include (e.g., ['.txt', '.md'])
        - csv_column (int, optional): Column index to use for CSV files (default is to join all columns)
        - csv_has_header (bool): Whether CSV files have a header row (default: True)
        - min_length (int): Minimum text length to include (default: 0, no filtering)
        - skip_duplicates (bool): Whether to skip duplicate texts (default: False)
        - recursive (bool): Whether to search directories recursively (default: False)
        - encoding (str, optional): Force specific encoding (default: auto-detect)
        - batch_size (int): Number of texts to yield at once (default: 100)
        """
        self.path = path
        self.extensions = extensions or ['.txt']
        self.csv_column = csv_column
        self.csv_has_header = csv_has_header
        self.min_length = min_length
        self.skip_duplicates = skip_duplicates
        self.recursive = recursive
        self.encoding = encoding
        self.batch_size = batch_size
        
        # For duplicate tracking
        self.seen_hashes: Set[str] = set()
        
        # File list for directory processing
        self.file_list: List[str] = []
        self.current_file_idx = 0
        
        # For CSV processing
        self.csv_file = None
        self.csv_reader = None
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Set up the iterator based on the provided path."""
        if os.path.isdir(self.path):
            # Get list of files to process
            if self.recursive:
                self.file_list = []
                for ext in self.extensions:
                    # Use rglob for recursive search
                    pattern = '**/*' + ext
                    matches = glob.glob(os.path.join(self.path, pattern), recursive=True)
                    self.file_list.extend(matches)
            else:
                self.file_list = []
                for ext in self.extensions:
                    pattern = '*' + ext
                    matches = glob.glob(os.path.join(self.path, pattern))
                    self.file_list.extend(matches)
                    
            # Sort for deterministic order
            self.file_list.sort()

            # If directory search yielded nothing, raise for caller
            if not self.file_list:
                raise ValueError(
                    f"No files with extensions {self.extensions} found in directory {self.path}"
                )
                
        elif os.path.isfile(self.path) and any(self.path.lower().endswith(ext) for ext in self.extensions):
            # Single text file
            self.file_list = [self.path]
            
        elif os.path.isfile(self.path) and self.path.lower().endswith('.csv'):
            # Process CSV file
            encoding = self._detect_encoding(self.path) if not self.encoding else self.encoding
            self.csv_file = open(self.path, 'r', encoding=encoding, newline='')
            self.csv_reader = csv.reader(self.csv_file)
            
            # Skip header if needed
            if self.csv_has_header:
                next(self.csv_reader, None)
        else:
            raise ValueError(f"The path {self.path} is not a supported file type or directory")
        
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect encoding of a file. Falls back to UTF-8 if detection fails.
        
        Parameters:
        - file_path (str): Path to the file
        
        Returns:
        - str: Detected encoding
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(min(1024 * 1024, os.path.getsize(file_path)))  # Read up to 1MB
            
            result = chardet.detect(raw_data)
            if result['confidence'] > 0.7:
                return result['encoding']
        except Exception:
            pass
        
        # Default to UTF-8 if detection fails or confidence is low
        return 'utf-8'
    
    def _get_text_hash(self, text: str) -> str:
        """
        Create a hash for a text string for duplicate detection.
        
        Parameters:
        - text (str): The text to hash
        
        Returns:
        - str: Hash of the text
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_duplicate(self, text: str) -> bool:
        """
        Check if a text is a duplicate.
        
        Parameters:
        - text (str): Text to check
        
        Returns:
        - bool: True if the text is a duplicate, False otherwise
        """
        if not self.skip_duplicates:
            return False
        
        text_hash = self._get_text_hash(text)
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def _process_text(self, text: str) -> Optional[str]:
        """
        Process a text string, applying filters.
        
        Parameters:
        - text (str): Text to process
        
        Returns:
        - Optional[str]: Processed text, or None if it should be skipped
        """
        # Skip empty texts
        if not text or not text.strip():
            return None
        
        # Skip short texts
        if len(text) < self.min_length:
            return None
        
        # Skip duplicates
        if self._is_duplicate(text):
            return None
        
        return text
    
    def _read_text_file(self, file_path: str) -> Optional[str]:
        """
        Read a text file with encoding detection.
        
        Parameters:
        - file_path (str): Path to the text file
        
        Returns:
        - Optional[str]: File content or None if there was an error
        """
        try:
            encoding = self._detect_encoding(file_path) if not self.encoding else self.encoding
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            # Try common encodings as fallback
            for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                if encoding != enc:  # Skip if we already tried this encoding
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            return f.read()
                    except Exception:
                        pass
            
            print(f"Warning: Could not read file {file_path}: {str(e)}")
            return None
    
    def _read_csv_batch(self) -> List[str]:
        """
        Read a batch of rows from the CSV file.
        
        Returns:
        - List[str]: Batch of processed texts
        """
        if not self.csv_reader:
            return []
        
        texts = []
        for _ in range(self.batch_size):
            try:
                row = next(self.csv_reader)
                
                # Extract text from specified column or join all columns
                if self.csv_column is not None and self.csv_column < len(row):
                    text = row[self.csv_column]
                else:
                    text = ','.join(row)
                
                processed_text = self._process_text(text)
                if processed_text:
                    texts.append(processed_text)
                    
            except StopIteration:
                # Close the file when we're done
                if self.csv_file:
                    self.csv_file.close()
                    self.csv_file = None
                    self.csv_reader = None
                break
            except Exception as e:
                print(f"Warning: Error processing CSV row: {str(e)}")
        
        return texts
    
    def _read_text_file_batch(self) -> List[str]:
        """
        Read a batch of text files.
        
        Returns:
        - List[str]: Batch of processed texts
        """
        texts = []
        
        while len(texts) < self.batch_size and self.current_file_idx < len(self.file_list):
            file_path = self.file_list[self.current_file_idx]
            self.current_file_idx += 1
            
            text = self._read_text_file(file_path)
            if text:
                processed_text = self._process_text(text)
                if processed_text:
                    texts.append(processed_text)
        
        return texts
    
    def __iter__(self):
        """Return self as iterator."""
        return self
    
    def __next__(self) -> List[str]:
        """
        Get the next batch of texts.
        
        Returns:
        - List[str]: Next batch of texts
        
        Raises:
        - StopIteration: When there are no more texts to read
        """
        batch = []
        
        # CSV file processing
        if self.csv_reader:
            batch = self._read_csv_batch()
        
        # Text file processing
        elif self.file_list:
            batch = self._read_text_file_batch()
        
        # If we got no texts and we're done with all files, stop iteration
        if not batch:
            raise StopIteration
        
        return batch
    
    def close(self):
        """Close any open files."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_reader = None


def load_data(file_path: str, 
              csv_column: int = None, 
              extensions: List[str] = None,
              recursive: bool = False,
              min_length: int = 0,
              skip_duplicates: bool = False) -> List[str]:
    """
    Load data from a file path, which can be either a directory containing text files
    or a CSV file.
    
    This function maintains backward compatibility with the original implementation
    while adding new features.
    
    Parameters:
    - file_path (str): Path to either a directory or a CSV file
    - csv_column (int, optional): Column index to use for CSV files (default is to join all columns)
    - extensions (List[str], optional): List of file extensions to include (default: ['.txt'])
    - recursive (bool): Whether to search directories recursively (default: False)
    - min_length (int): Minimum text length to include (default: 0, no filtering)
    - skip_duplicates (bool): Whether to skip duplicate texts (default: False)
    
    Returns:
    - list: List of text strings, where each string is either the content of a text file
            or a row from the CSV file
    """
    # Set default extensions
    if extensions is None:
        extensions = ['.txt']
    
    # Ensure all extensions start with a dot
    normalized_extensions = [ext if ext.startswith('.') else '.' + ext for ext in extensions]
    
    # Create iterator with a very large batch size to get all texts at once
    iterator = TextFileIterator(
        path=file_path,
        extensions=normalized_extensions,
        csv_column=csv_column,
        min_length=min_length,
        skip_duplicates=skip_duplicates,
        recursive=recursive,
        batch_size=1000000  # Very large to get all at once
    )
    
    # Get all texts in one batch
    try:
        all_texts = next(iterator)
        return all_texts
    except StopIteration:
        return []
    finally:
        iterator.close()


def get_file_iterator(file_path: str, 
                     batch_size: int = 100,
                     csv_column: int = None,
                     csv_has_header: bool = True,
                     extensions: List[str] = None,
                     recursive: bool = False,
                     min_length: int = 0,
                     skip_duplicates: bool = False,
                     encoding: str = None) -> TextFileIterator:
    """
    Create an iterator for reading text files or CSV files in batches.
    
    Parameters:
    - file_path (str): Path to either a directory or a CSV file
    - batch_size (int): Number of texts to yield at once (default: 100)
    - csv_column (int, optional): Column index to use for CSV files (default is to join all columns)
    - csv_has_header (bool): Whether CSV files have a header row (default: True)
    - extensions (List[str], optional): List of file extensions to include (default: ['.txt'])
    - recursive (bool): Whether to search directories recursively (default: False)
    - min_length (int): Minimum text length to include (default: 0, no filtering)
    - skip_duplicates (bool): Whether to skip duplicate texts (default: False)
    - encoding (str, optional): Force specific encoding (default: auto-detect)
    
    Returns:
    - TextFileIterator: Iterator for the specified files
    """
    # Set default extensions
    if extensions is None:
        extensions = ['.txt']
    
    # Ensure all extensions start with a dot
    normalized_extensions = [ext if ext.startswith('.') else '.' + ext for ext in extensions]
    
    return TextFileIterator(
        path=file_path,
        extensions=normalized_extensions,
        csv_column=csv_column,
        csv_has_header=csv_has_header,
        min_length=min_length,
        skip_duplicates=skip_duplicates,
        recursive=recursive,
        encoding=encoding,
        batch_size=batch_size
    ) 