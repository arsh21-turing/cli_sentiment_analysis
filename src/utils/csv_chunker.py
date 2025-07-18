import os
import csv
from typing import List, Dict, Any, Optional, Iterator, Tuple, Set, Union
# ---------------------------------------------------------------------------
# Optional heavy dependencies (pandas, tqdm)
# ---------------------------------------------------------------------------
# (importlib removed; handle pandas import gracefully)

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – provide lightweight fallback
    pd = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – create stub
    import types, sys

    def tqdm(iterable=None, *args, **kwargs):  # type: ignore
        if iterable is not None:
            return iterable
        # For non-iterable usage (like tqdm(total=100)), return a dummy object
        class DummyTqdm:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
        return DummyTqdm()

    _dummy = types.ModuleType("tqdm")
    _dummy.tqdm = tqdm  # type: ignore[attr-defined]
    sys.modules.setdefault("tqdm", _dummy)

import chardet
import codecs
from datetime import datetime
from io import StringIO

class CSVChunker:
    """Memory-efficient CSV chunking system that handles large input files."""
    
    def __init__(self,
                 file_path: str,
                 chunk_size: int = 10000,
                 text_columns: Optional[Union[int, str, List[Union[int, str]]]] = None,
                 delimiter: Optional[str] = None,
                 encoding: Optional[str] = None,
                 has_header: Optional[bool] = None,
                 sample_size: int = 1000,
                 validate_schema: bool = True,
                 min_length: int = 0,
                 skip_duplicates: bool = False):
        self.file_path = file_path
        self.chunk_size = chunk_size
        # Ensure text_columns is always a list
        if text_columns is not None:
            if isinstance(text_columns, (int, str)):
                self.text_columns = [text_columns]
            else:
                self.text_columns = list(text_columns)
        else:
            self.text_columns = None
        self.delimiter = delimiter
        self.encoding = encoding
        self.has_header = has_header
        self.sample_size = sample_size
        self.validate_schema = validate_schema
        self.min_length = min_length
        self.skip_duplicates = skip_duplicates
        
        # State tracking
        self.file_size = os.path.getsize(file_path)
        self.seen_texts: Set[str] = set()
        self.current_position = 0
        self.line_count = 0
        self.processed_size = 0
        
        # Analyze the CSV file
        self._analyze_csv()

    def _analyze_csv(self) -> None:
        """Analyze CSV file to detect properties."""
        # Read sample for detection
        with open(self.file_path, 'rb') as f:
            raw_sample = f.read(min(1024 * 1024, self.file_size))
        
        # Detect encoding
        if not self.encoding:
            result = chardet.detect(raw_sample)
            self.encoding = result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
        
        # Read sample lines
        sample_lines = []
        with codecs.open(self.file_path, 'r', encoding=self.encoding) as f:
            for _ in range(self.sample_size):
                line = f.readline()
                if not line:
                    break
                sample_lines.append(line)
        
        # Detect delimiter
        if not self.delimiter:
            self.delimiter = self._detect_delimiter(''.join(sample_lines))
        
        # Detect header
        if self.has_header is None:
            self.has_header = self._detect_header(''.join(sample_lines))
        
        # Parse sample to detect columns
        if pd is not None:
            try:
                df = pd.read_csv(
                    self.file_path,
                    nrows=self.sample_size,
                    sep=self.delimiter,
                    header=0 if self.has_header else None,
                    encoding=self.encoding,
                )
                self.columns = list(df.columns)

                # Only auto-detect text columns if not provided by user
                if self.text_columns is None:
                    self.text_columns = self._detect_text_columns(df)
            except Exception:
                self._fallback_sample_detection()
        else:
            # No pandas available
            self._fallback_sample_detection()

    def _fallback_sample_detection(self) -> None:
        """Simpler column detection when pandas is unavailable."""
        with codecs.open(self.file_path, 'r', encoding=self.encoding) as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            first_row = next(reader)
            self.columns = (
                first_row if self.has_header else [str(i) for i in range(len(first_row))]
            )
            # Only auto-detect text columns if not provided by user
            if self.text_columns is None:
                self.text_columns = [1] if len(first_row) > 1 else [0]

    def _detect_delimiter(self, sample: str) -> str:
        """Detect CSV delimiter."""
        delimiters = [',', ';', '\t', '|']
        best_delimiter = ','
        max_fields = 0
        for d in delimiters:
            counts = [line.count(d) for line in sample.splitlines() if line.strip()]
            if not counts:
                continue
            avg = sum(counts) / len(counts)
            if avg > max_fields:
                max_fields = avg
                best_delimiter = d
        return best_delimiter

    def _detect_header(self, sample: str) -> bool:
        """Detect if CSV has header row."""
        if pd is not None:
            try:
                df = pd.read_csv(StringIO(sample), sep=self.delimiter, nrows=5)
                first_row = df.columns
                return any(not str(col).isdigit() for col in first_row)
            except Exception:
                pass
        # Fallback: assume header if first line looks like column names
        lines = sample.splitlines()
        if not lines:
            return False
        first_line = lines[0]
        first_cells = first_line.split(self.delimiter)
        
        # Check if first row looks like column names (short, no numbers, common header words)
        header_indicators = ['id', 'text', 'category', 'score', 'name', 'title', 'description', 'label']
        first_cell_lower = first_cells[0].strip().lower() if first_cells else ""
        
        # If first cell looks like a header word, assume it's a header
        if any(indicator in first_cell_lower for indicator in header_indicators):
            return True
            
        # If all cells are short and don't contain numbers, likely a header
        all_short = all(len(cell.strip()) < 20 for cell in first_cells)
        no_numbers = all(not any(c.isdigit() for c in cell.strip()) for cell in first_cells)
        
        return all_short and no_numbers

    from typing import Any as _Any

    def _detect_text_columns(self, df: _Any) -> List[int]:
        """Detect columns containing text data."""
        if pd is None:
            return [1] if len(self.columns) > 1 else [0]
        text_cols: List[int] = []
        for i, col in enumerate(df.columns):
            if df[col].dtype == 'object':
                series = df[col].dropna().astype(str)
                if not series.empty and series.str.len().mean() > 5:
                    text_cols.append(i)
        return text_cols if text_cols else ([1] if len(df.columns) > 1 else [0])

    def __iter__(self) -> Iterator[List[str]]:
        """Iterate through CSV in chunks."""
        try:
            with codecs.open(self.file_path, 'r', encoding=self.encoding) as f:
                # Skip header if needed
                if self.has_header:
                    f.readline()
                    self.current_position = f.tell()
                
                # Process chunks
                current_chunk = []
                reader = csv.reader(f, delimiter=self.delimiter)
                
                for row in reader:
                    self.current_position = f.tell()
                    
                    # Extract text from specified columns
                    texts = []
                    for col_idx in self.text_columns:
                        if col_idx < len(row):
                            texts.append(row[col_idx])
                    
                    if texts:
                        text = ' '.join(texts)
                        
                        # Apply filters
                        if len(text) < self.min_length:
                            continue
                            
                        if self.skip_duplicates:
                            text_hash = hash(text)
                            if text_hash in self.seen_texts:
                                continue
                            self.seen_texts.add(text_hash)
                        
                        current_chunk.append(text)
                        
                        # Yield chunk when full
                        if len(current_chunk) >= self.chunk_size:
                            self.line_count += len(current_chunk)
                            self.processed_size = self.current_position
                            yield current_chunk
                            current_chunk = []
                
                # Yield remaining chunk
                if current_chunk:
                    self.line_count += len(current_chunk)
                    self.processed_size = self.current_position
                    yield current_chunk
                    
        except Exception as e:
            print(f"Error reading CSV: {e}")
            yield []

    def get_progress(self) -> Tuple[int, int, float]:
        """Get processing progress."""
        progress = (self.processed_size / self.file_size) * 100 if self.file_size > 0 else 0
        return self.processed_size, self.file_size, progress

def process_large_csv(
    file_path: str,
    analyzer,
    text_columns: Optional[Union[int, str, List[Union[int, str]]]] = None,
    batch_size: int = 1000,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    has_header: Optional[bool] = None,
    min_length: int = 0,
    skip_duplicates: bool = False,
    parallel: bool = False,
    max_workers: int = 4
) -> Dict[str, Any]:
    """Process large CSV with memory-efficient chunking."""
    from ..utils.batch import _process_sequential, _process_parallel, _compile_statistics
    
    start_time = datetime.now()
    
    # Create chunker
    chunker = CSVChunker(
        file_path=file_path,
        chunk_size=batch_size,
        text_columns=text_columns,
        delimiter=delimiter,
        encoding=encoding,
        has_header=has_header,
        min_length=min_length,
        skip_duplicates=skip_duplicates
    )
    
    # Process chunks
    all_results = []
    batch_count = 0
    total_texts = 0
    
    with tqdm(total=100, desc="CSV Progress", unit="%") as pbar:
        last_progress = 0
        
        for chunk in chunker:
            batch_count += 1
            total_texts += len(chunk)
            
            # Process batch
            if parallel and len(chunk) > 1:
                batch_results = _process_parallel(chunk, analyzer, max_workers)
            else:
                batch_results = _process_sequential(chunk, analyzer)
            
            all_results.extend(batch_results)
            
            # Update progress
            _, _, progress = chunker.get_progress()
            progress_change = int(progress) - last_progress
            if progress_change > 0:
                pbar.update(progress_change)
                last_progress = int(progress)
    
    # Calculate statistics
    stats = _compile_statistics(all_results)
    stats["processing_time"] = {
        "total_seconds": (datetime.now() - start_time).total_seconds(),
        "texts_per_second": total_texts / max((datetime.now() - start_time).total_seconds(), 1),
        "formatted": str(datetime.now() - start_time)
    }
    
    return {
        "source": file_path,
        "total_texts": total_texts,
        "batch_count": batch_count,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "individual_results": all_results,
        "statistics": stats
    } 