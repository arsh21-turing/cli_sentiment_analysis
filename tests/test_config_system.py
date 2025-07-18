"""
Tests for the configuration system.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

from src.config import Config, get_config


class TestConfigurationLoading:
    """Test configuration loading from various sources."""
    
    def test_default_configuration(self):
        """Test that default configuration loads correctly."""
        config = Config()
        
        # Test some default values
        assert config.get('models.groq.model') == 'llama2-70b-4096'
        assert config.get('thresholds.sentiment') == 0.5
        assert config.get('fallback.enabled') == False
        assert config.get('output.format') == 'text'
        assert config.get('preprocessing.remove_urls') == True
    
    def test_yaml_file_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
models:
  groq:
    model: mixtral-8x7b-32768
    timeout: 60
thresholds:
  sentiment: 0.8
  fallback: 0.3
fallback:
  enabled: true
  strategy: highest_confidence
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = Config(config_path=temp_path)
            
            # Test loaded values
            assert config.get('models.groq.model') == 'mixtral-8x7b-32768'
            assert config.get('models.groq.timeout') == 60
            assert config.get('thresholds.sentiment') == 0.8
            assert config.get('thresholds.fallback') == 0.3
            assert config.get('fallback.enabled') == True
            assert config.get('fallback.strategy') == 'highest_confidence'
            
            # Test that defaults are preserved for unspecified values
            assert config.get('thresholds.emotion') == 0.4  # default
            assert config.get('output.format') == 'text'  # default
            
        finally:
            os.unlink(temp_path)
    
    def test_json_file_loading(self):
        """Test loading configuration from JSON file."""
        json_content = {
            "models": {
                "groq": {
                    "model": "gemma-7b-it",
                    "cache": False
                }
            },
            "output": {
                "color": "always",
                "emoji": False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            temp_path = f.name
        
        try:
            config = Config(config_path=temp_path)
            
            # Test loaded values
            assert config.get('models.groq.model') == 'gemma-7b-it'
            assert config.get('models.groq.cache') == False
            assert config.get('output.color') == 'always'
            assert config.get('output.emoji') == False
            
            # Test defaults preserved
            assert config.get('models.groq.timeout') == 30  # default
            
        finally:
            os.unlink(temp_path)
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            'SENTIMENT_MODELS__GROQ__MODEL': 'claude-3-sonnet-20240229',
            'SENTIMENT_THRESHOLDS__SENTIMENT': '0.75',
            'SENTIMENT_FALLBACK__ENABLED': 'true',
            'SENTIMENT_OUTPUT__COLOR': 'never',
            'SENTIMENT_MODELS__GROQ__CACHE_SIZE': '200'
        }
        
        with patch.dict(os.environ, env_vars):
            config = Config()
            
            # Test env var values
            assert config.get('models.groq.model') == 'claude-3-sonnet-20240229'
            assert config.get('thresholds.sentiment') == 0.75
            assert config.get('fallback.enabled') == True
            assert config.get('output.color') == 'never'
            assert config.get('models.groq.cache_size') == 200
    
    def test_config_precedence(self):
        """Test that configuration sources have correct precedence."""
        # Create a config file
        yaml_content = """
models:
  groq:
    model: mixtral-8x7b-32768
thresholds:
  sentiment: 0.6
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Set environment variables
            env_vars = {
                'SENTIMENT_MODELS__GROQ__MODEL': 'gemma-7b-it',
                'SENTIMENT_THRESHOLDS__EMOTION': '0.5'
            }
            
            with patch.dict(os.environ, env_vars):
                config = Config(config_path=temp_path)
                
                # Environment should override file
                assert config.get('models.groq.model') == 'gemma-7b-it'
                
                # File should override defaults
                assert config.get('thresholds.sentiment') == 0.6
                
                # Environment should set new values
                assert config.get('thresholds.emotion') == 0.5
                
                # Defaults should be preserved where not overridden
                assert config.get('output.format') == 'text'
                
        finally:
            os.unlink(temp_path)


class TestConfigurationAPI:
    """Test configuration API methods."""
    
    def test_get_method(self):
        """Test the get method with paths and defaults."""
        config = Config()
        
        # Test existing path
        assert config.get('models.groq.model') == 'llama2-70b-4096'
        
        # Test non-existing path with default
        assert config.get('non.existing.path', 'default_value') == 'default_value'
        
        # Test non-existing path without default
        assert config.get('non.existing.path') is None
    
    def test_set_method(self):
        """Test the set method."""
        config = Config()
        
        # Set a new value
        config.set('test.path', 'test_value')
        assert config.get('test.path') == 'test_value'
        
        # Overwrite existing value
        config.set('models.groq.model', 'new_model')
        assert config.get('models.groq.model') == 'new_model'
    
    def test_specialized_getters(self):
        """Test specialized getter methods."""
        config = Config()
        
        # Test get_transformer_config
        transformer_config = config.get_transformer_config()
        assert isinstance(transformer_config, dict)
        assert 'sentiment_model' in transformer_config
        assert 'emotion_model' in transformer_config
        
        # Test get_groq_config
        groq_config = config.get_groq_config()
        assert isinstance(groq_config, dict)
        assert 'model' in groq_config
        assert 'timeout' in groq_config
        
        # Test get_thresholds
        thresholds = config.get_thresholds()
        assert isinstance(thresholds, dict)
        assert 'sentiment' in thresholds
        assert 'emotion' in thresholds
        
        # Test get_fallback_config
        fallback_config = config.get_fallback_config()
        assert isinstance(fallback_config, dict)
        assert 'enabled' in fallback_config
        assert 'strategy' in fallback_config
    
    def test_flattened_view(self):
        """Test getting flattened configuration view."""
        config = Config()
        flattened = config.get_flattened()
        
        assert isinstance(flattened, dict)
        assert 'models.groq.model' in flattened
        assert 'thresholds.sentiment' in flattened
        assert 'fallback.enabled' in flattened
        
        # Test that values match
        assert flattened['models.groq.model'] == config.get('models.groq.model')
    
    def test_config_sources_tracking(self):
        """Test that configuration sources are tracked correctly."""
        config = Config()
        
        # Set values from different sources
        config.set('test.default', 'value1', 'default')
        config.set('test.file', 'value2', 'file:test.yaml')
        config.set('test.env', 'value3', 'env:TEST_VAR')
        config.set('test.arg', 'value4', 'arg:test_arg')
        
        sources = config.get_sources()
        assert sources['test.default'] == 'default'
        assert sources['test.file'] == 'file:test.yaml'
        assert sources['test.env'] == 'env:TEST_VAR'
        assert sources['test.arg'] == 'arg:test_arg'


class TestConfigurationPersistence:
    """Test configuration saving and loading."""
    
    def test_save_yaml_format(self):
        """Test saving configuration in YAML format."""
        config = Config()
        
        # Modify some values
        config.set('models.groq.model', 'test_model')
        config.set('thresholds.sentiment', 0.9)
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            config.save_to_file(temp_path, format='yaml')
            
            # Load it back and verify
            loaded_config = Config(config_path=temp_path)
            assert loaded_config.get('models.groq.model') == 'test_model'
            assert loaded_config.get('thresholds.sentiment') == 0.9
            
        finally:
            os.unlink(temp_path)
    
    def test_save_json_format(self):
        """Test saving configuration in JSON format."""
        config = Config()
        
        # Modify some values
        config.set('models.groq.model', 'test_model')
        config.set('output.format', 'json')
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            config.save_to_file(temp_path, format='json')
            
            # Load it back and verify
            loaded_config = Config(config_path=temp_path)
            assert loaded_config.get('models.groq.model') == 'test_model'
            assert loaded_config.get('output.format') == 'json'
            
        finally:
            os.unlink(temp_path)


class TestArgumentIntegration:
    """Test integration with command line arguments."""
    
    def test_update_from_args(self):
        """Test updating configuration from command line arguments."""
        import argparse
        
        config = Config()
        
        # Create mock arguments
        args = argparse.Namespace()
        args.groq_model = 'test_model'
        args.sentiment_threshold = 0.85
        args.fallback_threshold = 0.25
        args.use_fallback = True
        args.no_color = True
        args.json = True
        args.quiet = True
        
        # Set None for arguments that shouldn't be processed
        args.no_fallback = None
        args.json_stream = None
        
        # Update config from args
        config.update_from_args(args)
        
        # Test that values were updated
        assert config.get('models.groq.model') == 'test_model'
        assert config.get('thresholds.sentiment') == 0.85
        assert config.get('thresholds.fallback') == 0.25
        assert config.get('fallback.enabled') == True
        assert config.get('output.color') == 'never'
        assert config.get('output.format') == 'json'
        assert config.get('output.quiet') == True
    
    def test_special_argument_handling(self):
        """Test special argument handling cases."""
        import argparse
        
        config = Config()
        
        # Test no_fallback flag
        args = argparse.Namespace()
        args.no_fallback = True
        args.use_fallback = None
        
        config.update_from_args(args)
        assert config.get('fallback.enabled') == False
        
        # Test json_stream flag
        config = Config()
        args = argparse.Namespace()
        args.json_stream = True
        args.quiet = None
        
        config.update_from_args(args)
        assert config.get('output.format') == 'json_stream'


class TestUtilityMethods:
    """Test utility methods of the configuration system."""
    
    def test_should_use_color(self):
        """Test color detection logic."""
        config = Config()
        
        # Test 'always' setting
        config.set('output.color', 'always')
        assert config.should_use_color() == True
        
        # Test 'never' setting
        config.set('output.color', 'never')
        assert config.should_use_color() == False
        
        # Test 'auto' setting (depends on TTY)
        config.set('output.color', 'auto')
        # Result depends on environment, just test it doesn't crash
        result = config.should_use_color()
        assert isinstance(result, bool)
    
    def test_is_fallback_enabled(self):
        """Test fallback enabled check."""
        config = Config()
        
        # Default should be False
        assert config.is_fallback_enabled() == False
        
        # Test enabling
        config.set('fallback.enabled', True)
        assert config.is_fallback_enabled() == True
        
        # Test disabling
        config.set('fallback.enabled', False)
        assert config.is_fallback_enabled() == False


class TestGlobalConfigInstance:
    """Test the global configuration instance management."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        # Clear any existing instance
        import src.config
        src.config._config = None
        
        # Get two instances
        config1 = get_config()
        config2 = get_config()
        
        # Should be the same object
        assert config1 is config2
    
    def test_get_config_reload(self):
        """Test reloading configuration."""
        # Clear any existing instance
        import src.config
        src.config._config = None
        
        config1 = get_config()
        config1.set('test.value', 'test1')
        
        # Reload should create new instance
        config2 = get_config(reload=True)
        
        # Should be different objects
        assert config1 is not config2
        
        # New instance shouldn't have the test value
        assert config2.get('test.value') is None


class TestErrorHandling:
    """Test error handling in configuration system."""
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration files."""
        # Test non-existent file
        config = Config(config_path='non_existent_file.yaml')
        # Should not crash, just use defaults
        assert config.get('models.groq.model') == 'llama2-70b-4096'
    
    def test_malformed_yaml(self):
        """Test handling of malformed YAML files."""
        malformed_yaml = """
models:
  groq:
    model: test
  invalid: [unclosed list
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(malformed_yaml)
            temp_path = f.name
        
        try:
            # Should not crash, just log error and use defaults
            config = Config(config_path=temp_path)
            assert config.get('models.groq.model') == 'llama2-70b-4096'  # default
            
        finally:
            os.unlink(temp_path)
    
    def test_missing_yaml_support(self):
        """Test behavior when YAML support is not available."""
        with patch('src.config.HAS_YAML', False):
            # Should still work with JSON files
            json_content = '{"models": {"groq": {"model": "test_model"}}}'
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(json_content)
                temp_path = f.name
            
            try:
                config = Config(config_path=temp_path)
                assert config.get('models.groq.model') == 'test_model'
                
            finally:
                os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__]) 