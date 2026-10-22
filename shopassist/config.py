"""Paths and config loading."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "golden"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"


def load_config(name: str) -> dict:
    """Load config by short name (v1_baseline) or by path (config/v1_baseline.yaml)."""
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = ROOT / "config" / f"{name}.yaml"
    elif not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_path"] = str(path.relative_to(ROOT))
    return cfg


def golden_path(version: str) -> Path:
    return GOLDEN_DIR / f"golden_{version}.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_golden(version: str) -> list[dict]:
    path = golden_path(version)
    if not path.exists():
        raise SystemExit(f"Golden dataset not found: {path}. Did you run 'golden release {version}'?")
    return load_jsonl(path)
