"""
Tests for input validation and error handling.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.utils.validation import (
    ValidationError, validate_text_input, validate_batch_file, validate_file_format,
    validate_export_path, validate_model_name, validate_threshold, check_api_key_availability,
    validate_args, print_error, handle_exception, get_byte_size, word_count,
    file_line_generator, check_dependencies_or_warn
)


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error_with_message_only(self):
        """Test ValidationError with message only."""
        error = ValidationError("Test error")
        assert error.message == "Test error"
        assert error.suggestion is None
        assert str(error) == "Test error"
    
    def test_validation_error_with_suggestion(self):
        """Test ValidationError with message and suggestion."""
        error = ValidationError("Test error", "Try this fix")
        assert error.message == "Test error"
        assert error.suggestion == "Try this fix"
        assert str(error) == "Test error"


class TestTextValidation:
    """Test text input validation."""
    
    def test_validate_empty_text(self):
        """Test validation of empty text."""
        with pytest.raises(ValidationError) as exc_info:
            validate_text_input("")
        assert "Empty input text" in exc_info.value.message
    
    def test_validate_whitespace_only_text(self):
        """Test validation of whitespace-only text."""
        with pytest.raises(ValidationError) as exc_info:
            validate_text_input("   \n\t  ")
        assert "Empty input text" in exc_info.value.message
    
    @patch('src.utils.validation.get_config')
    def test_validate_text_too_short(self, mock_config):
        """Test validation of text below minimum length."""
        mock_config.return_value.get.side_effect = lambda key, default: {
            'advanced.min_length': 10,
            'advanced.max_length': 512
        }.get(key, default)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_text_input("short")
        assert "too short" in exc_info.value.message
    
    @patch('src.utils.validation.get_config')
    def test_validate_text_truncation(self, mock_config):
        """Test text truncation when too long."""
        mock_config.return_value.get.side_effect = lambda key, default: {
            'advanced.min_length': 1,
            'advanced.max_length': 10
        }.get(key, default)
        
        long_text = "This is a very long text that should be truncated"
        result = validate_text_input(long_text)
        assert len(result) == 10
        assert result == long_text[:10]
    
    @patch('src.utils.validation.get_config')
    def test_validate_normal_text(self, mock_config):
        """Test validation of normal text."""
        mock_config.return_value.get.side_effect = lambda key, default: {
            'advanced.min_length': 1,
            'advanced.max_length': 512
        }.get(key, default)
        
        text = "This is a normal text for analysis."
        result = validate_text_input(text)
        assert result == text.strip()


class TestFileValidation:
    """Test file validation."""
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_file("/nonexistent/file.txt")
        assert "File not found" in exc_info.value.message
    
    def test_validate_directory_as_file(self):
        """Test validation when directory is passed as file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValidationError) as exc_info:
                validate_batch_file(tmpdir)
            assert "Not a file" in exc_info.value.message
    
    def test_validate_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            file_path = f.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                validate_batch_file(file_path)
            assert "empty" in exc_info.value.message.lower()
        finally:
            os.unlink(file_path)
    
    @patch('src.utils.validation.get_config')
    def test_validate_file_too_large(self, mock_config):
        """Test validation of file that's too large."""
        mock_config.return_value.get.side_effect = lambda key, default: {
            'advanced.max_file_size': 100  # 100 bytes
        }.get(key, default)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("x" * 200)  # Write 200 bytes
            file_path = f.name
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                validate_batch_file(file_path)
            assert "too large" in exc_info.value.message
        finally:
            os.unlink(file_path)
    
    @patch('src.utils.validation.get_config')
    def test_validate_valid_file(self, mock_config):
        """Test validation of valid file."""
        mock_config.return_value.get.side_effect = lambda key, default: {
            'advanced.max_file_size': 10 * 1024 * 1024  # 10MB
        }.get(key, default)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("This is test content\nSecond line\n")
            file_path = f.name
        
        try:
            result = validate_batch_file(file_path)
            assert result == file_path
        finally:
            os.unlink(file_path)


class TestFileFormatValidation:
    """Test file format validation."""
    
    def test_validate_allowed_format(self):
        """Test validation of allowed file format."""
        result = validate_file_format("test.txt", ["txt", "csv", "json"])
        assert result == "test.txt"
    
    def test_validate_disallowed_format(self):
        """Test validation of disallowed file format."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_format("test.pdf", ["txt", "csv", "json"])
        assert "Unsupported file format" in exc_info.value.message
    
    def test_validate_case_insensitive_format(self):
        """Test case-insensitive format validation."""
        result = validate_file_format("test.TXT", ["txt", "csv"])
        assert result == "test.TXT"


class TestExportPathValidation:
    """Test export path validation."""
    
    def test_validate_export_to_existing_directory(self):
        """Test export path validation with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "results.csv")
            result = validate_export_path(export_path)
            assert result == export_path
    
    def test_validate_export_to_new_directory(self):
        """Test export path validation with new directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_folder")
            export_path = os.path.join(new_dir, "results.csv")
            result = validate_export_path(export_path)
            assert result == export_path
            assert os.path.exists(new_dir)


class TestModelNameValidation:
    """Test model name validation."""
    
    def test_validate_empty_model_name(self):
        """Test validation of empty model name."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("", "sentiment")
        assert "Empty sentiment model name" in exc_info.value.message
    
    def test_validate_local_model_path_exists(self):
        """Test validation of existing local model path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_model_name(tmpdir, "sentiment")
            assert result == tmpdir
    
    def test_validate_local_model_path_missing(self):
        """Test validation of missing local model path."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("/nonexistent/model", "sentiment")
        assert "Model path not found" in exc_info.value.message
    
    @patch('src.utils.validation.GroqModel')
    def test_validate_groq_model_valid(self, mock_groq):
        """Test validation of valid Groq model."""
        mock_groq.AVAILABLE_MODELS = {"llama-3.1-8b": {}, "mixtral-8x7b": {}}
        
        result = validate_model_name("llama-3.1-8b", "groq")
        assert result == "llama-3.1-8b"
    
    @patch('src.utils.validation.GroqModel')
    def test_validate_groq_model_invalid(self, mock_groq):
        """Test validation of invalid Groq model."""
        mock_groq.AVAILABLE_MODELS = {"llama-3.1-8b": {}, "mixtral-8x7b": {}}
        
        with pytest.raises(ValidationError) as exc_info:
            validate_model_name("invalid-model-123!", "groq")
        assert "Invalid Groq model name" in exc_info.value.message


class TestThresholdValidation:
    """Test threshold validation."""
    
    def test_validate_valid_threshold(self):
        """Test validation of valid threshold."""
        result = validate_threshold("0.5", "sentiment")
        assert result == 0.5
    
    def test_validate_threshold_string_number(self):
        """Test validation of threshold as string number."""
        result = validate_threshold("0.85", "emotion")
        assert result == 0.85
    
    def test_validate_threshold_non_numeric(self):
        """Test validation of non-numeric threshold."""
        with pytest.raises(ValidationError) as exc_info:
            validate_threshold("not_a_number", "sentiment")
        assert "Invalid sentiment threshold" in exc_info.value.message
    
    def test_validate_threshold_below_range(self):
        """Test validation of threshold below valid range."""
        with pytest.raises(ValidationError) as exc_info:
            validate_threshold("-0.1", "emotion")
        assert "must be between 0.0 and 1.0" in exc_info.value.message
    
    def test_validate_threshold_above_range(self):
        """Test validation of threshold above valid range."""
        with pytest.raises(ValidationError) as exc_info:
            validate_threshold("1.5", "fallback")
        assert "must be between 0.0 and 1.0" in exc_info.value.message


class TestAPIKeyValidation:
    """Test API key validation."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_check_missing_api_key(self):
        """Test check for missing API key."""
        with pytest.raises(ValidationError) as exc_info:
            check_api_key_availability("Test Service", "TEST_API_KEY")
        assert "No API key found" in exc_info.value.message
    
    @patch.dict(os.environ, {"TEST_API_KEY": "test_key_value"})
    def test_check_existing_api_key(self):
        """Test check for existing API key."""
        # Should not raise exception
        check_api_key_availability("Test Service", "TEST_API_KEY")


class TestArgumentValidation:
    """Test command-line argument validation."""
    
    def test_validate_conflicting_input_methods(self):
        """Test validation of conflicting input methods."""
        args = MagicMock()
        args.text = "sample text"
        args.file = "sample.txt"
        args.batch = None
        args.interactive = False
        args.show_settings = False
        args.reset_settings = False
        args.show_config = False
        args.show_config_sources = False
        args.save_config_file = None
        
        with pytest.raises(ValidationError) as exc_info:
            validate_args(args)
        assert "Multiple input methods" in exc_info.value.message
    
    def test_validate_missing_input_method(self):
        """Test validation of missing input method."""
        args = MagicMock()
        args.text = None
        args.file = None
        args.batch = None
        args.interactive = False
        args.show_settings = False
        args.reset_settings = False
        args.show_config = False
        args.show_config_sources = False
        args.save_config_file = None
        
        with pytest.raises(ValidationError) as exc_info:
            validate_args(args)
        assert "No input method specified" in exc_info.value.message


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_byte_size_bytes(self):
        """Test byte size calculation for bytes."""
        result = get_byte_size("hello")
        assert result == "5 B"
    
    def test_get_byte_size_kb(self):
        """Test byte size calculation for kilobytes."""
        text = "x" * 1500  # 1.5 KB
        result = get_byte_size(text)
        assert "1.5 KB" in result
    
    def test_get_byte_size_mb(self):
        """Test byte size calculation for megabytes."""
        text = "x" * (2 * 1024 * 1024)  # 2 MB
        result = get_byte_size(text)
        assert "2.0 MB" in result
    
    def test_word_count(self):
        """Test word counting."""
        assert word_count("hello world") == 2
        assert word_count("one") == 1
        assert word_count("") == 1  # split() on empty string returns [""]
        assert word_count("  multiple   spaces   ") == 2
    
    def test_check_dependencies_available(self):
        """Test dependency checking with available packages."""
        result = check_dependencies_or_warn(["os", "sys"], "test feature")
        assert result is True
    
    def test_check_dependencies_missing(self):
        """Test dependency checking with missing packages."""
        result = check_dependencies_or_warn(["nonexistent_package"], "test feature")
        assert result is False


class TestFileLineGenerator:
    """Test file line generator."""
    
    def test_file_line_generator_basic(self):
        """Test basic file line generation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line 1\nline 2\nline 3\n")
            file_path = f.name
        
        try:
            lines = list(file_line_generator(file_path))
            assert lines == ["line 1", "line 2", "line 3"]
        finally:
            os.unlink(file_path)
    
    def test_file_line_generator_max_lines(self):
        """Test file line generation with max lines limit."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line 1\nline 2\nline 3\nline 4\nline 5\n")
            file_path = f.name
        
        try:
            lines = list(file_line_generator(file_path, max_lines=3))
            assert lines == ["line 1", "line 2", "line 3"]
        finally:
            os.unlink(file_path)
    
    def test_file_line_generator_skip_empty(self):
        """Test file line generation skipping empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line 1\n\nline 3\n   \nline 5\n")
            file_path = f.name
        
        try:
            lines = list(file_line_generator(file_path, skip_empty=True))
            assert lines == ["line 1", "line 3", "line 5"]
        finally:
            os.unlink(file_path)


class TestExceptionHandling:
    """Test exception handling."""
    
    def test_handle_validation_error(self):
        """Test handling of ValidationError."""
        exc = ValidationError("Test error", "Test suggestion")
        result = handle_exception(exc, debug_mode=False)
        assert result == 1
    
    def test_handle_keyboard_interrupt(self):
        """Test handling of KeyboardInterrupt."""
        exc = KeyboardInterrupt()
        result = handle_exception(exc, debug_mode=False)
        assert result == 130
    
    def test_handle_memory_error(self):
        """Test handling of MemoryError."""
        exc = MemoryError("Out of memory")
        result = handle_exception(exc, debug_mode=False)
        assert result == 3
    
    def test_handle_permission_error(self):
        """Test handling of PermissionError."""
        exc = PermissionError("Access denied")
        result = handle_exception(exc, debug_mode=False)
        assert result == 4
    
    def test_handle_generic_exception_debug_mode(self):
        """Test handling of generic exception in debug mode."""
        exc = RuntimeError("Generic error")
        with patch('src.utils.validation.traceback.print_exc') as mock_traceback:
            result = handle_exception(exc, debug_mode=True)
            assert result == 1
            mock_traceback.assert_called_once()
    
    def test_handle_generic_exception_normal_mode(self):
        """Test handling of generic exception in normal mode."""
        exc = RuntimeError("Generic error")
        with patch('src.utils.validation.logger.exception') as mock_logger:
            result = handle_exception(exc, debug_mode=False)
            assert result == 1
            mock_logger.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__]) 