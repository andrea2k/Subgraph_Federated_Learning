"""Commands for training, graph construction, and thesis reporting."""

from __future__ import annotations
import argparse
import json
import os
import sys
from .config import experiment_config
from .paths import absolute

METHODS = (
    "fully_local",
    "centralized",
    "fedavg",
    "fedprox",
    "gcflplus",
    "fedala",
    "apple",
    "cwfedavg_task",
    "fedptha",
    "fedptha_postala",
    "learned_only_postala",
    "support_only_postala",
    "oracle_physical",
    "hierarchical",
    "specialist",
)


def parser():
    result = argparse.ArgumentParser(
        prog="fedptha", description="Run the FedPTHA thesis experiments."
    )
    result.add_argument(
        "--project-root", help="Project/data root; defaults to the checkout."
    )
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Train one method using JSON configuration."
    )
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument(
        "--benchmark", choices=("five_client", "twenty_client"), required=True
    )
    train.add_argument("--seeds", type=int, nargs="+")
    train.add_argument("--q", type=float, nargs="+")
    train.add_argument("--rounds", type=int)
    train.add_argument(
        "--output-dir", help="Run group root; defaults to results/runs/default."
    )
    train.add_argument("--device", choices=("cpu", "cuda"))
    train.add_argument(
        "--dry-run", action="store_true", help="Show configuration without training."
    )
    generate = commands.add_parser(
        "generate-data", help="Construct a separate benchmark dataset."
    )
    generate.add_argument(
        "--benchmark", choices=("five_client", "twenty_client"), required=True
    )
    generate.add_argument("--output-dir")
    generate.add_argument("--overwrite", action="store_true")
    report = commands.add_parser(
        "reproduce", help="Rebuild thesis tables and figures from frozen results."
    )
    report.add_argument("--output-dir", default="results/generated")
    report.add_argument("--figures-dir", default="figures/generated")
    report.add_argument(
        "--start-at", help="Resume at a stage using existing intermediate outputs."
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.project_root:
        os.environ["FEDPTHA_PROJECT_ROOT"] = str(absolute(args.project_root))
    try:
        if args.command == "train":
            if args.dry_run:
                settings = experiment_config(args.method, args.benchmark)
                settings["requested"] = {
                    key: getattr(args, key)
                    for key in ("seeds", "q", "rounds", "output_dir", "device")
                }
                print(json.dumps(settings, indent=2))
            else:
                from .training.experiment import run

                run(
                    args.method,
                    args.benchmark,
                    seeds=args.seeds,
                    q=args.q,
                    rounds=args.rounds,
                    output_dir=args.output_dir,
                    device=args.device,
                )
        elif args.command == "generate-data":
            from .data.generate import generate

            generate(args.benchmark, args.output_dir, overwrite=args.overwrite)
        else:
            os.environ["FEDPTHA_OUTPUT_DIR"] = str(absolute(args.output_dir))
            os.environ["FEDPTHA_FIGURES_DIR"] = str(absolute(args.figures_dir))
            os.environ.setdefault(
                "MPLCONFIGDIR", str(absolute("results/.cache/matplotlib"))
            )
            from .reporting.pipeline import reproduce

            reproduce(start_at=args.start_at)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
