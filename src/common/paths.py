from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = os.getenv("LLM_EXPERIMENT", "exp1")

LEGACY_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / DEFAULT_EXPERIMENT
NEW_DATA_DIR = PROJECT_ROOT / "data" / "experiments" / DEFAULT_EXPERIMENT
NEW_PROMPTS_DIR = PROJECT_ROOT / "prompts" / "experiments" / DEFAULT_EXPERIMENT
NEW_OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments" / DEFAULT_EXPERIMENT

ENV_EXPERIMENT_ROOT = os.getenv("LLM_EXPERIMENT_ROOT")


# 兼容两种目录模式：
# 1) 新架构（默认）：data/experiments/<exp>, prompts/experiments/<exp>, output/experiments/<exp>
# 2) 旧架构（兼容）：experiments/<exp>/{data,prompts,results}
if ENV_EXPERIMENT_ROOT:
    EXPERIMENT_ROOT = Path(ENV_EXPERIMENT_ROOT)
    DATA_DIR = Path(os.getenv("LLM_DATA_DIR", str(EXPERIMENT_ROOT / "data")))
    PROMPTS_DIR = Path(os.getenv("LLM_PROMPTS_DIR", str(EXPERIMENT_ROOT / "prompts")))
    RESULTS_DIR = Path(os.getenv("LLM_RESULTS_DIR", str(EXPERIMENT_ROOT / "results")))
else:
    DATA_DIR = Path(
        os.getenv(
            "LLM_DATA_DIR",
            str(NEW_DATA_DIR if NEW_DATA_DIR.exists() else LEGACY_EXPERIMENT_ROOT / "data"),
        )
    )
    PROMPTS_DIR = Path(
        os.getenv(
            "LLM_PROMPTS_DIR",
            str(NEW_PROMPTS_DIR if NEW_PROMPTS_DIR.exists() else LEGACY_EXPERIMENT_ROOT / "prompts"),
        )
    )
    RESULTS_DIR = Path(
        os.getenv(
            "LLM_RESULTS_DIR",
            str(NEW_OUTPUT_DIR if NEW_OUTPUT_DIR.exists() else LEGACY_EXPERIMENT_ROOT / "results"),
        )
    )
    EXPERIMENT_ROOT = Path(os.getenv("LLM_EXPERIMENT_ROOT", str(DATA_DIR)))


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
