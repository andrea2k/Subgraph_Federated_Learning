"""Stable public names; frozen experiment identifiers are never rewritten."""

METHOD_NAMES = {
    "fully_local": "Fully Local",
    "centralized": "Global-Centralized",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedala": "FedALA",
    "gcflplus": "GCFL+",
    "apple": "APPLE",
    "cwfedavg_task": "cwFedAvg-Task",
    "support_simplex": "FedPTHA",
    "support_simplex_postala": "FedPTHA + PostALA",
    "learned_simplex_postala": "Learned source weights only + PostALA",
    "support_only_postala": "Positive-label distribution only + PostALA",
}
PUBLIC_METHODS = {
    "support_simplex": "fedptha",
    "support_simplex_postala": "fedptha_postala",
    "learned_simplex_postala": "learned_only_postala",
}
