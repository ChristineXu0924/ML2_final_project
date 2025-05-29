# src/config.py
import yaml
from pathlib import Path

def load_config(config_name: str = "project_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
