"""Repository and artifact locations, independent of the caller's directory."""

from __future__ import annotations
import os
from pathlib import Path


def project_root() -> Path:
    configured = os.environ.get("FEDPTHA_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs").is_dir():
        return candidate
    return Path.cwd().resolve()


def absolute(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def frozen_root() -> Path:
    return absolute(os.environ.get("FEDPTHA_FROZEN_ROOT", "results/frozen"))


def generated_root() -> Path:
    return absolute(os.environ.get("FEDPTHA_OUTPUT_DIR", "results/generated"))


def reference_root() -> Path:
    return frozen_root() / "reference"


def figures_root() -> Path:
    return absolute(os.environ.get("FEDPTHA_FIGURES_DIR", "figures/generated"))
