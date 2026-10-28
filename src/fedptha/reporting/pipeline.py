"""Rebuild derived analysis in dependency order, export paper-facing tables."""

from __future__ import annotations
import json
import subprocess
import sys
from fedptha.paths import generated_root, figures_root, project_root, frozen_root

STAGES = [
    ("extract_results", []),
    ("summarize_results", []),
    ("extra_experiments", []),
    ("protocol_analysis", []),
    ("ablations", []),
    ("collaboration", []),
    ("collaboration_comparison", []),
    ("payload_five", ["--benchmark-setup", "five_client"]),
    ("payload_twenty", ["--benchmark-setup", "twenty_client"]),
    ("regional_payload", []),
    ("communication", []),
    ("communication_performance", []),
    ("appendix_tables", []),
    ("diagnostic_tradeoff", []),
    ("selected_figures", []),
    ("figure_ablation", []),
    ("figure_main_all", []),
    ("figure_protocols_all", []),
    ("figure_communication_all", []),
]


def reproduce(*, start_at=None):
    output = generated_root()
    figures = figures_root()
    for destination in (output, figures):
        if destination == frozen_root() or frozen_root() in destination.parents:
            raise ValueError("Regenerated outputs must be outside results/frozen/.")
    if start_at in (None, "extract_results"):
        missing = [
            name
            for name in ("main", "additions", "cwfedavg")
            if not (frozen_root() / name).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"Frozen experiment archive is required at {frozen_root()} "
                f"(missing: {', '.join(missing)}). See docs/reproduction.md."
            )
    for directory in (output / "tables", output / "atlases", output / "logs", figures):
        directory.mkdir(parents=True, exist_ok=True)
    stages = list(STAGES)
    if start_at:
        names = [name for name, _ in stages]
        if start_at not in names:
            raise ValueError(f"Unknown stage {start_at}; choose from {names}")
        stages = stages[names.index(start_at) :]
    completed = []
    for name, arguments in stages:
        module = "fedptha.reporting." + (
            "payload"
            if name.startswith("payload_")
            else "regional_payload" if name == "regional_payload" else "tasks." + name
        )
        log = output / "logs" / f"{name}.log"
        print(f"Rebuilding {name} ...", flush=True)
        with log.open("w") as stream:
            process = subprocess.run(
                [sys.executable, "-u", "-m", module, *arguments],
                cwd=project_root(),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if process.returncode:
            tail = "\n".join(log.read_text().splitlines()[-25:])
            raise RuntimeError(
                f"{name} failed (exit {process.returncode}). See {log}\n{tail}"
            )
        completed.append(name)
    from .export import export_paper_results

    export_paper_results()
    (output / "pipeline.json").write_text(
        json.dumps({"stages": completed}, indent=2) + "\n"
    )
    print(f"Figures: {figures}\nTables: {output}")
