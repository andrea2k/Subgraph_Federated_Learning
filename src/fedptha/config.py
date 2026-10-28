"""Read the shared training, benchmark, and method JSON configurations."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import absolute


def read_config(path: str | Path) -> dict:
    with absolute(path).open() as stream:
        return json.load(stream)


def experiment_config(method: str, benchmark: str) -> dict:
    cfg = read_config("configs/training.json")
    method_cfg = read_config(f"configs/methods/{method}.json")
    benchmark_cfg = read_config(f"configs/benchmarks/{benchmark}.json")
    cfg.update(method_cfg)
    cfg["parameters"].update(benchmark_cfg.get("method_overrides", {}).get(method, {}))
    cfg["method"] = method
    cfg["benchmark"] = benchmark
    cfg["benchmark_config"] = benchmark_cfg
    return cfg
