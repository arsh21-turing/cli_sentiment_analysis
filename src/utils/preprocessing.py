# src/utils/preprocessing.py
import re
import html
import unicodedata
import langid
from typing import List, Optional, Union, Literal

class TextPreprocessor:
    """
    A comprehensive text preprocessing utility for NLP tasks.
    
    Performs text normalization, cleaning, and tokenization with configurable options.
    """
    
    def __init__(
        self, 
        remove_urls: bool = True,
        remove_html: bool = True,
        fix_encoding: bool = True,
        handle_emojis: Literal['keep', 'remove', 'replace'] = 'keep',
        lowercase: bool = True
    ):
        """
        Initialize TextPreprocessor with configurable preprocessing options.
        
        Args:
            remove_urls: Whether to remove URLs from text
            remove_html: Whether to remove HTML tags from text
            fix_encoding: Whether to fix Unicode encoding issues
            handle_emojis: How to handle emojis ('keep', 'remove', or 'replace')
            lowercase: Whether to convert text to lowercase
        """
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.fix_encoding = fix_encoding
        self.handle_emojis = handle_emojis
        self.lowercase = lowercase
        
        # URL regex pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # HTML tag regex pattern
        self.html_pattern = re.compile(r'<.*?>')
        
        # Emoji regex pattern (simplified)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+"
        )
        
    def preprocess(self, text: str) -> str:
        """
        Apply the complete preprocessing pipeline to the input text.
        
        Args:
            text: The input text to preprocess
            
        Returns:
            The preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Fix encoding issues if enabled
        if self.fix_encoding:
            text = self.fix_unicode(text)
            
        # Remove URLs if enabled
        if self.remove_urls:
            text = self.remove_urls_from_text(text)
            
        # Remove HTML tags if enabled
        if self.remove_html:
            text = self.remove_html_tags(text)
            
        # Process emojis according to settings
        if self.handle_emojis in ['remove', 'replace']:
            text = self.process_emojis(text)
            
        # Convert to lowercase if enabled
        if self.lowercase:
            text = text.lower()
            
        # Normalize whitespace
        text = self.normalize_whitespace(text)
            
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: The input text
            
        Returns:
            ISO language code of the detected language
        """
        lang, _ = langid.classify(text)
        return lang
    
    def remove_special_chars(self, text: str) -> str:
        """
        Remove special characters from text while preserving whitespace.
        
        Args:
            text: The input text
            
        Returns:
            Text with special characters removed
        """
        return re.sub(r'[^\w\s]', '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace by replacing multiple spaces with a single space.
        
        Args:
            text: The input text
            
        Returns:
            Text with normalized whitespace
        """
        return re.sub(r'\s+', ' ', text)
    
    def remove_urls_from_text(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: The input text
            
        Returns:
            Text with URLs removed
        """
        return self.url_pattern.sub('', text)
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text and decode HTML entities.
        
        Args:
            text: The input text
            
        Returns:
            Text with HTML tags removed and entities decoded
        """
        text = self.html_pattern.sub('', text)
        return html.unescape(text)
    
    def fix_unicode(self, text: str) -> str:
        """
        Fix Unicode encoding issues and normalize character forms.
        
        Args:
            text: The input text
            
        Returns:
            Text with fixed encoding
        """
        return unicodedata.normalize('NFKC', text)
    
    def process_emojis(self, text: str) -> str:
        """
        Process emojis according to settings.
        
        Args:
            text: The input text
            
        Returns:
            Text with emojis processed
        """
        if self.handle_emojis == 'remove':
            return self.emoji_pattern.sub('', text)
        elif self.handle_emojis == 'replace':
            # Replace emojis with their text description (simplified)
            # In a real implementation, you'd use a mapping
            return self.emoji_pattern.sub(' [emoji] ', text)
        else:
            return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        This is a simple whitespace tokenization. For more advanced tokenization,
        use a specialized tokenizer.
        
        Args:
            text: The input text
            
        Returns:
            List of tokens
        """
        return text.split() 