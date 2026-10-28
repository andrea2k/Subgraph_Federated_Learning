"""Reader-facing results with final paper method and protocol names."""

from .names import METHOD_NAMES, PUBLIC_METHODS
from fedptha.paths import generated_root


def export_paper_results():
    import pandas as pd

    source = generated_root() / "tables/main_ac_summary.csv"
    frame = pd.read_csv(source)
    frame["method_display"] = frame["method"].map(METHOD_NAMES)
    frame["method"] = frame["method"].replace(PUBLIC_METHODS)
    frame.insert(3, "protocol", "AC")
    frame.to_csv(generated_root() / "tables/paper_results.csv", index=False)
