"""Select the best available- and complete-validation states by Macro PR-AUC."""

import math


def is_better_selection_value(current, best):
    if current is None or math.isnan(float(current)):
        return False
    return float(current) > float(best)
