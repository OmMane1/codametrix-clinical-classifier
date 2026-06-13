"""Build clean_medical_text.csv from the Medical Transcriptions dataset.

Source: Kaggle ``tboyle10/medicaltranscriptions`` (``mtsamples.csv``), which has
real orthopedics cases (unlike the Medical Abstracts corpus where orthopedics
was empty).

This regenerates the SAME contract the encoder + TF-IDF tracks already consume
(``encoder_text`` + ``hackathon_label``, plus ``clean_text`` for classical
models), so nothing downstream changes. Cleaning is delegated to the team's
``data_cleaning.py`` so encoder/classical text is normalised identically.

Specialty -> hackathon class mapping is strict: only clearly-matching
specialties map to a named class; every other specialty falls into ``other``.

Usage:
    # auto-download via kagglehub (no local file needed):
    python scripts/build_transcriptions_dataset.py

    # or point at a local mtsamples.csv:
    python scripts/build_transcriptions_dataset.py --input path/to/mtsamples.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make the repo-root ``data_cleaning`` importable when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_cleaning import clean_for_classical_model, clean_for_encoder  # noqa: E402

# Hackathon class order (index == hackathon_label). Must match
# encoder.config.DEFAULT_LABEL_NAMES.
HACKATHON_CLASS_TO_LABEL = {
    "cardiology": 0,
    "neurology": 1,
    "orthopedics": 2,
    "gastroenterology": 3,
    "other": 4,
}

# Strict mapping from (stripped) mtsamples ``medical_specialty`` to a hackathon
# class. Anything not listed here -> "other".
SPECIALTY_TO_HACKATHON = {
    "Cardiovascular / Pulmonary": "cardiology",
    "Neurology": "neurology",
    "Neurosurgery": "neurology",        # nervous-system; flip to "other" to be stricter
    "Orthopedic": "orthopedics",
    "Gastroenterology": "gastroenterology",
}


SPECIFIC_CLASSES = {"cardiology", "neurology", "orthopedics", "gastroenterology"}


def map_specialty(specialty: str) -> str:
    return SPECIALTY_TO_HACKATHON.get(str(specialty).strip(), "other")


def resolve_duplicates(df: pd.DataFrame):
    """Collapse each distinct ``encoder_text`` to one row with the best label.

    mtsamples files the same transcription under multiple specialties, so after
    mapping a single text can carry several hackathon classes. We resolve this
    label noise deterministically:

    * one class (possibly repeated)          -> keep one row, that class
    * {other, exactly one specific}          -> keep the SPECIFIC class
      (the "other" copy came from a generic filing like Surgery/Consult; the
      specific specialty is the informative label)
    * {two or more specific specialties}     -> DROP (true ambiguity)

    This prevents naive keep-first dedup from deleting the minority classes, and
    removes only genuinely ambiguous texts. Returns ``(resolved_df, n_dropped)``.
    """

    def target_for(classes: set):
        specifics = classes - {"other"}
        if len(specifics) >= 2:
            return None  # ambiguous -> drop
        if len(specifics) == 1:
            return next(iter(specifics))
        return "other"

    classes_per_text = df.groupby("encoder_text")["hackathon_classification"].agg(set)
    targets = classes_per_text.apply(target_for)
    n_dropped_ambiguous = int(targets.isna().sum())

    df = df.copy()
    df["_target"] = df["encoder_text"].map(targets)
    df = df[df["_target"].notna()]
    # Keep only the row whose class equals the resolved target, then one per text.
    df = df[df["hackathon_classification"] == df["_target"]]
    df = df.drop_duplicates(subset="encoder_text").drop(columns=["_target"])
    return df.reset_index(drop=True), n_dropped_ambiguous


def build_dataframe(input_csv: str | Path) -> pd.DataFrame:
    """Load mtsamples.csv and produce the contract DataFrame."""
    df = pd.read_csv(input_csv)
    required = {"transcription", "medical_specialty"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{input_csv} missing column(s) {sorted(missing)}")

    # Transcription is the rich clinical note; drop rows without it.
    df = df.dropna(subset=["transcription", "medical_specialty"]).copy()

    df["source_specialty"] = df["medical_specialty"].str.strip()
    df["hackathon_classification"] = df["source_specialty"].map(map_specialty)
    df["hackathon_label"] = df["hackathon_classification"].map(HACKATHON_CLASS_TO_LABEL)

    df["raw_text"] = df["transcription"].astype(str)
    df["encoder_text"] = df["transcription"].apply(clean_for_encoder)
    df["clean_text"] = df["transcription"].apply(clean_for_classical_model)
    df["text_length"] = df["encoder_text"].str.len()
    df["token_count"] = df["encoder_text"].str.split().str.len()

    # Drop rows that cleaned down to nothing.
    df = df[df["encoder_text"].str.len() > 0].reset_index(drop=True)

    keep = [
        "source_specialty",
        "hackathon_classification",
        "hackathon_label",
        "raw_text",
        "encoder_text",
        "clean_text",
        "text_length",
        "token_count",
    ]
    return df[keep]


def summarize(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}")
    print("class distribution (hackathon_label):")
    counts = df["hackathon_classification"].value_counts()
    for name, label in sorted(HACKATHON_CLASS_TO_LABEL.items(), key=lambda kv: kv[1]):
        print(f"  {label} {name:18s}: {int(counts.get(name, 0))}")
    dup = int(df.duplicated("encoder_text").sum())
    print(f"duplicate encoder_text rows: {dup}")


def resolve_input(input_arg: str | None) -> Path:
    if input_arg:
        return Path(input_arg)
    import kagglehub

    path = Path(kagglehub.dataset_download("tboyle10/medicaltranscriptions"))
    csv = path / "mtsamples.csv"
    if not csv.exists():
        matches = list(path.rglob("mtsamples.csv"))
        if not matches:
            raise FileNotFoundError(f"mtsamples.csv not found under {path}")
        csv = matches[0]
    return csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to mtsamples.csv (default: kagglehub download).")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "clean_medical_text.csv"),
        help="Output CSV path (default: repo-root clean_medical_text.csv).",
    )
    args = parser.parse_args()

    input_csv = resolve_input(args.input)
    print(f"Using input: {input_csv}")
    df = build_dataframe(input_csv)
    before = len(df)
    df, n_ambiguous = resolve_duplicates(df)
    print(
        f"Resolved duplicates/collisions: {before} -> {len(df)} rows "
        f"(dropped {n_ambiguous} ambiguous multi-specialty texts)."
    )
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    summarize(df)


if __name__ == "__main__":
    main()
