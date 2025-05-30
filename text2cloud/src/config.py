"""Configuration loader for project YAML settings."""

from pathlib import Path
import yaml

def load_config(config_name: str = "project_config.yaml") -> dict:
    """Load YAML configuration file from the project's config folder."""
    config_path = Path(__file__).resolve().parent.parent / "config" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as file_handle:
        return yaml.safe_load(file_handle)
