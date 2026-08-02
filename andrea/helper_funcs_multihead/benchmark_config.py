from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkPaths:
    setup: str
    select_subset_path: str
    select_subset: str

    @property
    def selected_subsets_csv_path(self) -> str:
        return f"./andrea/{self.select_subset_path}/{self.select_subset}.csv"

    @property
    def registry_csv_path(self) -> str:
        return f"./andrea/{self.select_subset_path}/cluster_generation_parameters.csv"


_ALIASES = {
    "5": "five_client",
    "5client": "five_client",
    "5-client": "five_client",
    "five": "five_client",
    "five_client": "five_client",
    "five-client": "five_client",
    "20": "twenty_client",
    "20client": "twenty_client",
    "20-client": "twenty_client",
    "twenty": "twenty_client",
    "twenty_client": "twenty_client",
    "twenty-client": "twenty_client",
}

_DEFAULT_PATHS = {
    "five_client": ("clustering_q_label_heterogeneity", "selected_subset"),
    "twenty_client": ("clustering_planted_community_q", "selected_subset"),
}


def normalize_benchmark_setup(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _ALIASES:
        allowed = ", ".join(sorted(_DEFAULT_PATHS))
        raise ValueError(
            f"Unknown BENCHMARK_SETUP={value!r}. Use one of: {allowed} "
            "(aliases 5client and 20client are also accepted)."
        )
    return _ALIASES[key]


def resolve_benchmark_paths(default_setup: str = "five_client") -> BenchmarkPaths:
    """Resolve one common benchmark switch for every active runner.

    Primary switch:
        BENCHMARK_SETUP=five_client
        BENCHMARK_SETUP=twenty_client

    SELECT_SUBSET_PATH and SELECT_SUBSET remain optional explicit overrides for
    smoke tests or future manifests.
    """
    setup = normalize_benchmark_setup(
        os.environ.get("BENCHMARK_SETUP", default_setup)
    )
    default_path, default_subset = _DEFAULT_PATHS[setup]
    select_subset_path = os.environ.get("SELECT_SUBSET_PATH", default_path).strip()
    select_subset = os.environ.get("SELECT_SUBSET", default_subset).strip()

    if not select_subset_path:
        raise ValueError("SELECT_SUBSET_PATH resolved to an empty value.")
    if not select_subset:
        raise ValueError("SELECT_SUBSET resolved to an empty value.")

    return BenchmarkPaths(
        setup=setup,
        select_subset_path=select_subset_path,
        select_subset=select_subset,
    )
