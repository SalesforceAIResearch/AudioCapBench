#!/usr/bin/env python3
"""
Configuration and credentials handling for AudioCapBench.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_credentials(creds_path: Optional[str] = None) -> bool:
    """
    Load credentials from a .env file into environment variables.

    Supports formats:
        export KEY="value"
        export KEY=value
        KEY=value

    Args:
        creds_path: Path to credentials file. If None, searches for
                    credentials.env in project root.

    Returns:
        True if file was loaded, False otherwise.
    """
    if creds_path is None:
        # Search upward from this file for credentials.env
        search = Path(__file__).parent.parent / "credentials.env"
        if search.exists():
            creds_path = str(search)
        else:
            return False

    creds_file = Path(creds_path)
    if not creds_file.exists():
        return False

    with open(creds_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove 'export ' prefix
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value

    return True


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_config(
    config_path: Optional[str] = None,
    credentials_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load full configuration, merging YAML config with defaults.

    Args:
        config_path: Path to YAML config. If None, uses configs/default.yaml.
        credentials_path: Path to credentials.env file.

    Returns:
        Configuration dictionary.
    """
    # Load credentials into env
    load_credentials(credentials_path)

    # Load YAML config
    if config_path is None:
        default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if default_path.exists():
            config_path = str(default_path)

    if config_path and Path(config_path).exists():
        config = load_yaml_config(config_path)
    else:
        config = {}

    return config
