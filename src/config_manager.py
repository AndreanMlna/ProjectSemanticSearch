"""
Configuration Management
Centralized configuration handling for different environments
"""

import yaml
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    Manage application configuration from YAML/JSON file
    Supports hierarchical access using dot notation
    
    Example:
        config = ConfigManager()
        model_path = config.get("model.path")
        batch_size = config.get("training.batch_size")
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ConfigManager
        
        Args:
            config_path: Path to config file (YAML or JSON)
        
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config file format not supported
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from file
        Supports YAML and JSON formats
        
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If file format not supported
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        file_extension = Path(self.config_path).suffix.lower()
        
        try:
            if file_extension == '.yaml' or file_extension == '.yml':
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            elif file_extension == '.json':
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {file_extension}")
        except Exception as e:
            raise ValueError(f"Failed to load config file: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., "model.path", "training.batch_size")
            default: Default value if key not found
        
        Returns:
            Configuration value or default if not found
        
        Example:
            >>> config = ConfigManager()
            >>> config.get("model.path", "default/path")
            'output/minilm-dokumen-arsip-boosted'
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
                if value is default:
                    return default
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire section of configuration
        
        Args:
            section: Section name (e.g., "model", "training")
        
        Returns:
            Section dictionary or empty dict if not found
        
        Example:
            >>> config = ConfigManager()
            >>> config.get_section("training")
            {'batch_size': 16, 'epochs': 4, ...}
        """
        return self.config.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value (runtime only, doesn't persist to file)
        
        Args:
            key: Configuration key
            value: Value to set
        
        Example:
            >>> config = ConfigManager()
            >>> config.set("model.path", "new/path")
        """
        keys = key.split('.')
        # Anotasi eksplisit Dict[str, Any] agar IDE tahu tipe config
        # tetap Dict di setiap iterasi, bukan Any
        current: Dict[str, Any] = self.config

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

    def save_to_file(self, filepath: str = None) -> None:
        """
        Save current configuration to file

        Args:
            filepath: Path to save to (default: original config_path)

        Raises:
            IOError: If save fails
        """
        save_path = filepath or self.config_path

        try:
            file_extension = Path(save_path).suffix.lower()

            if file_extension == '.yaml' or file_extension == '.yml':
                with open(save_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif file_extension == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {file_extension}")
        except Exception as e:
            raise IOError(f"Failed to save config: {str(e)}")

    def reload(self) -> None:
        """Reload configuration from file"""
        self._load_config()

    def __repr__(self) -> str:
        return f"ConfigManager(path={self.config_path})"

    def __str__(self) -> str:
        return f"ConfigManager with {len(self.config)} sections"


# Global config instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get global ConfigManager instance (singleton pattern)

    Returns:
        ConfigManager instance

    Example:
        >>> config = get_config()
        >>> model_path = config.get("model.path")
    """
    global _config_instance

    if _config_instance is None:
        config_path = os.getenv("CONFIG_PATH", "config.yaml")
        _config_instance = ConfigManager(config_path)

    return _config_instance


def reset_config() -> None:
    """Reset global ConfigManager instance (useful for testing)"""
    global _config_instance
    _config_instance = None