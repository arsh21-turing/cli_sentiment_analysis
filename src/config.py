"""
Configuration system for the sentiment analysis tool.

This module provides a comprehensive configuration system that allows users
to customize the behavior of the application through a configuration file.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)

# Try to import yaml, but make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

class Config:
    """
    Comprehensive configuration system for the sentiment analysis tool.
    
    This class manages all configuration settings for the application, including
    model selection, thresholds, fallback behavior, output formats, and API settings.
    It can load settings from a configuration file and merge them with command-line
    arguments and environment variables.
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        # Model configuration
        "models": {
            # Primary transformer model settings
            "transformer": {
                "sentiment_model": "nlptown/bert-base-multilingual-uncased-sentiment",
                "emotion_model": "bhadresh-savani/distilbert-base-uncased-emotion",
                "local_model_path": None,  # Path to local model files
                "device": "auto"  # "auto", "cpu", "cuda", or "mps"
            },
            # Groq API settings
            "groq": {
                "model": "llama2-70b-4096",  # Default Groq model
                "api_key": None,  # Will be loaded from environment by default
                "timeout": 30,  # Seconds
                "max_retries": 3,
                "cache": True,  # Enable response caching
                "cache_size": 100  # Maximum number of cached responses
            }
        },
        # Confidence thresholds
        "thresholds": {
            "sentiment": 0.5,
            "emotion": 0.4,
            "fallback": 0.35,  # Threshold for triggering fallback
            # Multi-level thresholds
            "sentiment_levels": {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            },
            "emotion_levels": {
                "high": 0.7,
                "medium": 0.5,
                "low": 0.3
            }
        },
        # Fallback system configuration
        "fallback": {
            "enabled": False,  # Disabled by default
            "always_use": False,  # If True, always use fallback regardless of confidence
            "strategy": "weighted",  # Options: "weighted", "highest_confidence", "primary_first", "fallback_first"
            "show_details": False,  # Show fallback process details in output
            "weighted_primary_factor": 0.7,  # Factor for primary model in weighted strategy
            "conflict_threshold": 0.1  # Threshold for detecting conflicting emotions
        },
        # Output formatting
        "output": {
            "format": "text",  # Options: "text", "json", "json_stream"
            "color": "auto",  # Options: "auto", "always", "never"
            "quiet": False,  # Suppress informational messages
            "summary_only": False,  # Show only summary line
            "show_probabilities": False,  # Show raw probability distributions
            "emoji": True,  # Use emoji in text output
            "detailed": True  # Show detailed descriptions
        },
        # Preprocessing options
        "preprocessing": {
            "remove_urls": True,
            "remove_html": True,
            "fix_encoding": True,
            "handle_emojis": "keep",  # Options: "keep", "remove", "replace"
            "lowercase": True
        },
        # Logging configuration
        "logging": {
            "level": "INFO",  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            "file": None,  # Log file path, None for console only
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        # Advanced settings
        "advanced": {
            "batch_size": 16,  # Batch size for processing multiple texts
            "use_cuda": "auto",  # Options: "auto", "yes", "no"
            "use_mps": "auto",  # Options: "auto", "yes", "no" (Apple Silicon)
            "export_format": "csv",  # Default export format
            "temp_dir": None,  # Temporary directory for exports
            "cache_dir": None,  # Cache directory for models
            "max_length": 512  # Maximum sequence length for transformer models
        }
    }
    
    # Configuration file name options (in order of preference)
    CONFIG_FILE_OPTIONS = [
        ".sentimentrc",
        "sentiment.yaml",
        "sentiment.yml",
        "sentiment.json"
    ]
    
    # Environment variable prefix
    ENV_PREFIX = "SENTIMENT_"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration system.
        
        Args:
            config_path: Optional path to a configuration file.
                If not provided, will search for config files in standard locations.
        """
        # Initialize with default configuration
        import copy
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Track sources of configuration values for debugging
        self._sources = {}
        
        # Load configuration from file (if available)
        config_file = self._find_config_file(config_path)
        if config_file:
            self._load_from_file(config_file)
        
        # Load configuration from environment variables
        self._load_from_env()
        
        # Initialize logging based on configuration
        self._setup_logging()
    
    def _find_config_file(self, config_path: Optional[str] = None) -> Optional[str]:
        """
        Find the configuration file to use.
        
        Args:
            config_path: Optional explicit path to a config file
            
        Returns:
            Path to the config file, or None if no valid config file found
        """
        # If explicit path provided, use it
        if config_path and os.path.isfile(config_path):
            logger.debug(f"Using specified configuration file: {config_path}")
            return config_path
        
        # Search in current directory and home directory
        search_paths = [
            os.getcwd(),  # Current directory
            os.path.expanduser("~")  # Home directory
        ]
        
        for path in search_paths:
            for filename in self.CONFIG_FILE_OPTIONS:
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path):
                    logger.debug(f"Found configuration file: {file_path}")
                    return file_path
        
        logger.debug("No configuration file found, using default settings")
        return None
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            # Determine file format based on extension
            ext = os.path.splitext(config_path)[1].lower()
            
            if ext in ['.yaml', '.yml'] and HAS_YAML:
                with open(config_path, 'r') as file:
                    file_config = yaml.safe_load(file) or {}
            elif ext in ['.json']:
                with open(config_path, 'r') as file:
                    file_config = json.load(file)
            else:
                # For other formats (like .sentimentrc), try JSON first, then YAML
                try:
                    with open(config_path, 'r') as file:
                        file_config = json.load(file)
                except json.JSONDecodeError:
                    if HAS_YAML:
                        with open(config_path, 'r') as file:
                            file_config = yaml.safe_load(file) or {}
                    else:
                        logger.warning(f"Cannot parse {config_path}: YAML support not available")
                        return
            
            # Update configuration with values from file
            self._update_config_recursive(self.config, file_config, f"file:{config_path}")
            logger.debug(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with SENTIMENT_ and use double underscores
        to represent nesting, e.g., SENTIMENT_MODELS__GROQ__API_KEY.
        """
        # Get all environment variables with the correct prefix
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(self.ENV_PREFIX)}
        
        for key, value in env_vars.items():
            # Remove prefix and split into parts
            config_path = key[len(self.ENV_PREFIX):].lower()
            parts = config_path.split('__')
            
            # Convert value to appropriate type
            typed_value = self._parse_env_value(value)
            
            # Update config with this value
            self._set_config_value(parts, typed_value, source=f"env:{key}")
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse an environment variable value into the appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Parsed value as the appropriate type
        """
        # Try to parse as JSON first (for complex types)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Simple type conversion
            if value.lower() in ['true', 'yes', 'y', '1']:
                return True
            elif value.lower() in ['false', 'no', 'n', '0']:
                return False
            elif value.lower() in ['none', 'null']:
                return None
            
            # Try to parse as a number
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                # Just return as string if all else fails
                return value
    
    def _update_config_recursive(self, target: Dict, source: Dict, source_name: str, path: List[str] = None) -> None:
        """
        Update configuration dictionary recursively with values from another dictionary.
        
        Args:
            target: Target dictionary to update (in-place)
            source: Source dictionary with new values
            source_name: Source description for tracking
            path: Current path in the configuration (for tracking sources)
        """
        path = path or []
        
        for key, value in source.items():
            current_path = path + [key]
            
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursively update nested dictionary
                self._update_config_recursive(target[key], value, source_name, current_path)
            else:
                # Update leaf value
                target[key] = value
                self._sources['.'.join(current_path)] = source_name
    
    def _set_config_value(self, path_parts: List[str], value: Any, source: str) -> None:
        """
        Set a value in the configuration at the specified path.
        
        Args:
            path_parts: List of path components
            value: Value to set
            source: Source of the configuration value
        """
        # Navigate to the correct nested dictionary
        current = self.config
        for i, part in enumerate(path_parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Convert a leaf node to a dictionary if needed
                current[part] = {}
                
            current = current[part]
        
        # Set the value
        current[path_parts[-1]] = value
        
        # Record the source
        self._sources['.'.join(path_parts)] = source
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its path.
        
        Args:
            path: Dot-separated path to the configuration value
            default: Default value to return if the path doesn't exist
            
        Returns:
            Configuration value, or default if not found
        """
        parts = path.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, path: str, value: Any, source: str = "runtime") -> None:
        """
        Set a configuration value by its path.
        
        Args:
            path: Dot-separated path to the configuration value
            value: Value to set
            source: Source of the configuration value
        """
        parts = path.split('.')
        self._set_config_value(parts, value, source)
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """
        Update configuration based on command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Define mapping from argument names to config paths
        arg_mapping = {
            # Model settings
            'sentiment_model': 'models.transformer.sentiment_model',
            'emotion_model': 'models.transformer.emotion_model',
            'local_model_path': 'models.transformer.local_model_path',
            'device': 'models.transformer.device',
            'groq_model': 'models.groq.model',
            'groq_api_key': 'models.groq.api_key',
            
            # Threshold settings
            'sentiment_threshold': 'thresholds.sentiment',
            'emotion_threshold': 'thresholds.emotion',
            'fallback_threshold': 'thresholds.fallback',
            
            # Fallback settings
            'use_fallback': 'fallback.enabled',
            'no_fallback': None,  # Special case handled below
            'always_fallback': 'fallback.always_use',
            'show_fallback_details': 'fallback.show_details',
            'fallback_strategy': 'fallback.strategy',
            
            # Output settings
            'no_color': None,  # Special case handled below
            'color': 'output.color',
            'quiet': 'output.quiet',
            'summary_only': 'output.summary_only',
            'json': None,  # Special case handled below
            'json_stream': None,  # Special case handled below
            'probabilities': 'output.show_probabilities',
            
            # Other settings
            'log_level': 'logging.level',
            'log_file': 'logging.file',
            'batch_size': 'advanced.batch_size',
            'export_format': 'advanced.export_format',
        }
        
        # Update config from arguments
        for arg_name, arg_value in vars(args).items():
            # Skip None values (not specified in command line)
            if arg_value is None:
                continue
            
            # Handle special cases
            if arg_name == 'no_fallback' and arg_value:
                self.set('fallback.enabled', False, source=f"arg:{arg_name}")
            
            elif arg_name == 'no_color' and arg_value:
                self.set('output.color', 'never', source=f"arg:{arg_name}")
                
            elif arg_name == 'json' and arg_value:
                self.set('output.format', 'json', source=f"arg:{arg_name}")
                
            elif arg_name == 'json_stream' and arg_value:
                self.set('output.format', 'json_stream', source=f"arg:{arg_name}")
                
            # Handle normal mappings
            elif arg_name in arg_mapping and arg_mapping[arg_name]:
                self.set(arg_mapping[arg_name], arg_value, source=f"arg:{arg_name}")
        
        # Apply any additional logic based on argument combinations
        if self.get('output.format') in ['json', 'json_stream'] and hasattr(args, 'quiet') and args.quiet is None:
            # Automatically enable quiet mode for JSON output unless explicitly disabled
            self.set('output.quiet', True, source="auto:json_format")
    
    def save_to_file(self, file_path: str, format: str = 'yaml') -> None:
        """
        Save current configuration to a file.
        
        Args:
            file_path: Path to save the configuration to
            format: Format to save in ('yaml' or 'json')
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            if format.lower() == 'yaml' and HAS_YAML:
                with open(file_path, 'w') as file:
                    yaml.safe_dump(self.config, file, default_flow_style=False, sort_keys=False)
            else:  # json
                with open(file_path, 'w') as file:
                    json.dump(self.config, file, indent=2)
                    
            logger.info(f"Configuration saved to {file_path}")
        
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
    
    def get_sources(self) -> Dict[str, str]:
        """
        Get the sources of all configuration values.
        
        Returns:
            Dictionary mapping configuration paths to their sources
        """
        return self._sources.copy()
    
    def _setup_logging(self) -> None:
        """Configure logging based on the current configuration."""
        log_config = self.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('file')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Convert string level to numeric level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
        
        # File handler (if configured)
        if log_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(logging.Formatter(log_format))
                handlers.append(file_handler)
            except Exception as e:
                print(f"Warning: Could not set up log file {log_file}: {str(e)}")
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Add our handlers
        for handler in handlers:
            root_logger.addHandler(handler)
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            A deep copy of the current configuration
        """
        import copy
        return copy.deepcopy(self.config)
    
    def get_flattened(self) -> Dict[str, Any]:
        """
        Get a flattened view of the configuration (dot notation).
        
        Returns:
            Dictionary with dot-separated keys for all configuration values
        """
        result = {}
        self._flatten_dict(self.config, result)
        return result
    
    def _flatten_dict(self, d: Dict[str, Any], result: Dict[str, Any], prefix: str = '') -> None:
        """
        Helper method to flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            result: Dictionary to store results in
            prefix: Current key prefix
        """
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            
            if isinstance(v, dict):
                self._flatten_dict(v, result, key)
            else:
                result[key] = v
    
    def get_transformer_config(self) -> Dict[str, Any]:
        """
        Get configuration specific to the transformer model.
        
        Returns:
            Dictionary with transformer model configuration
        """
        return self.get('models.transformer', {})
    
    def get_groq_config(self) -> Dict[str, Any]:
        """
        Get configuration specific to the Groq API.
        
        Returns:
            Dictionary with Groq API configuration
        """
        return self.get('models.groq', {})
    
    def get_fallback_config(self) -> Dict[str, Any]:
        """
        Get configuration specific to the fallback system.
        
        Returns:
            Dictionary with fallback system configuration
        """
        return self.get('fallback', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get configuration specific to output formatting.
        
        Returns:
            Dictionary with output formatting configuration
        """
        return self.get('output', {})
    
    def get_thresholds(self) -> Dict[str, Any]:
        """
        Get all threshold configuration values.
        
        Returns:
            Dictionary with threshold configuration
        """
        return self.get('thresholds', {})
    
    def is_fallback_enabled(self) -> bool:
        """
        Check if the fallback system is enabled.
        
        Returns:
            True if fallback is enabled, False otherwise
        """
        return bool(self.get('fallback.enabled', False))
    
    def should_use_color(self) -> bool:
        """
        Determine if color should be used for output.
        
        Returns:
            True if color should be used, False otherwise
        """
        color_setting = self.get('output.color', 'auto')
        
        if color_setting == 'always':
            return True
        elif color_setting == 'never':
            return False
        else:  # auto
            # Check if stdout is a TTY
            import sys
            return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


# Global instance for easy access
_config = None

def get_config(reload: bool = False, config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        reload: Whether to force reloading the configuration
        config_path: Optional path to a configuration file
        
    Returns:
        Global Config instance
    """
    global _config
    if _config is None or reload:
        _config = Config(config_path)
    return _config 