from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = os.getenv("LLM_EXPERIMENT", "exp1")
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / DEFAULT_EXPERIMENT
EXPERIMENT_ROOT = Path(os.getenv("LLM_EXPERIMENT_ROOT", str(DEFAULT_EXPERIMENT_ROOT)))

DATA_DIR = EXPERIMENT_ROOT / "data"
PROMPTS_DIR = EXPERIMENT_ROOT / "prompts"
RESULTS_DIR = EXPERIMENT_ROOT / "results"


def project_root() -> Path:
    return PROJECT_ROOT


def experiment_root() -> Path:
    return EXPERIMENT_ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def prompt_path(*parts: str) -> Path:
    return PROMPTS_DIR.joinpath(*parts)


def results_path(*parts: str) -> Path:
    return RESULTS_DIR.joinpath(*parts)
