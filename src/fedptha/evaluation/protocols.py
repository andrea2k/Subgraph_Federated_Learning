"""Final thesis protocol names and their historical CSV encodings."""

PROTOCOLS = {
    "AA": ("best_realistic_visible", "available", "available"),
    "AC": ("best_realistic_selection_oracle", "available", "complete"),
    "CC": ("best_oracle_full", "complete", "complete"),
}
LEGACY_NAMES = {"VV": "AA", "VO": "AC", "OO": "CC"}


def normalize_protocol(value):
    value = LEGACY_NAMES.get(str(value).upper(), str(value).upper())
    if value not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {value}")
    return value
