"""Tests for the mtsamples -> clean_medical_text.csv builder.

Focus on the collision-aware dedup, which is the subtle, correctness-critical
part (naive dedup silently deletes minority classes on this dataset).
"""

import sys
from pathlib import Path

import pandas as pd

# The builder lives in scripts/ at the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_transcriptions_dataset import map_specialty, resolve_duplicates


def _df(rows):
    """rows: list of (encoder_text, hackathon_classification)."""
    return pd.DataFrame(
        {
            "encoder_text": [r[0] for r in rows],
            "hackathon_classification": [r[1] for r in rows],
        }
    )


def test_map_specialty_strips_and_maps_known():
    assert map_specialty(" Orthopedic") == "orthopedics"
    assert map_specialty("Cardiovascular / Pulmonary") == "cardiology"
    assert map_specialty(" Neurosurgery") == "neurology"


def test_map_specialty_unknown_falls_to_other():
    assert map_specialty(" Radiology") == "other"
    assert map_specialty(" Dentistry") == "other"


def test_resolve_keeps_single_label_text():
    df = _df([("note A", "cardiology"), ("note A", "cardiology")])
    out, dropped = resolve_duplicates(df)
    assert dropped == 0
    assert len(out) == 1
    assert out.iloc[0]["hackathon_classification"] == "cardiology"


def test_resolve_prefers_specific_over_other():
    # same text filed under generic "other" and specific "orthopedics"
    df = _df([("note B", "other"), ("note B", "orthopedics")])
    out, dropped = resolve_duplicates(df)
    assert dropped == 0
    assert len(out) == 1
    assert out.iloc[0]["hackathon_classification"] == "orthopedics"


def test_resolve_drops_two_specific_ambiguity():
    df = _df([("note C", "cardiology"), ("note C", "neurology")])
    out, dropped = resolve_duplicates(df)
    assert dropped == 1
    assert len(out) == 0


def test_resolve_mixed_batch():
    df = _df(
        [
            ("a", "cardiology"),            # single -> cardiology
            ("b", "other"),
            ("b", "gastroenterology"),      # other+specific -> gastroenterology
            ("c", "neurology"),
            ("c", "orthopedics"),           # 2 specifics -> drop
            ("d", "other"),                 # single -> other
        ]
    )
    out, dropped = resolve_duplicates(df)
    assert dropped == 1  # "c"
    got = dict(zip(out["encoder_text"], out["hackathon_classification"]))
    assert got == {"a": "cardiology", "b": "gastroenterology", "d": "other"}
