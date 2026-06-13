"""TF-IDF baseline for clinical note specialty classification.

Public API:
    build_pipeline   -> sklearn Pipeline (TF-IDF features + classifier)
    load_dataset     -> (texts, labels) from .csv / .dat
"""
from .pipeline import build_pipeline
from .data import load_dataset

__all__ = ["build_pipeline", "load_dataset"]
