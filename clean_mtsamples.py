"""Build cleaned 5-class training CSVs from the raw MTSamples transcriptions.

Self-contained (no project deps) so this data branch reproduces on its own.

Input : data/raw/mtsamples.csv   (Kaggle: tboyle10/medicaltranscriptions)
Output: data/mt_specialty.csv    (recommended) and data/mt_strict.csv

Why this isn't a one-liner: MTSamples is MULTI-LABELED. The same note text is
filed under several specialties (e.g. an echo note under both
"Cardiovascular / Pulmonary" and "Radiology"). We resolve each unique note to a
single hackathon label by PRIORITY — a target specialty beats the generic
"Other"/document-type buckets. See DATA_CARD.md for the full rationale.

Run:  python clean_mtsamples.py
"""
from __future__ import annotations

import pandas as pd

# The four target specialties -> hackathon class names. Everything else -> Other.
TARGET_MAP = {
    "Cardiovascular / Pulmonary": "Cardiology",
    "Neurology": "Neurology",
    "Orthopedic": "Orthopedics",
    "Gastroenterology": "Gastroenterology",
}
TARGETS = set(TARGET_MAP.values())

# Document-TYPE buckets — not specialties; they span every body system, so
# they're label-noise for a specialty classifier. "specialty" mode drops them.
DOCTYPE_BUCKETS = {
    "Surgery", "Consult - History and Phy.", "Radiology", "General Medicine",
    "SOAP / Chart / Progress Notes", "Discharge Summary", "Emergency Room Reports",
    "Office Notes", "Letters", "IME-QME-Work Comp etc.", "Lab Medicine - Pathology",
    "Autopsy",
}


def build(df: pd.DataFrame, other_mode: str):
    """other_mode: 'strict' (non-target -> Other) or 'specialty' (drop doc-type buckets)."""
    stats = {"conflicts": 0}

    def resolve(specs):
        hack = {TARGET_MAP.get(s, "__doc__" if s in DOCTYPE_BUCKETS else "Other")
                for s in specs}
        tgt = hack & TARGETS
        if len(tgt) == 1:
            return next(iter(tgt))
        if len(tgt) > 1:               # ambiguous across target specialties -> drop
            stats["conflicts"] += 1
            return None
        if other_mode == "specialty":  # keep as Other only if a real specialty
            return "Other" if hack - {"__doc__"} else None
        return "Other"                 # strict: everything non-target is Other

    labels = df.groupby("text")["spec"].apply(set).apply(resolve).dropna()
    out = labels.reset_index().rename(columns={"spec": "label"})[["text", "label"]]
    return out, stats


def main():
    df = pd.read_csv("data/raw/mtsamples.csv")
    df["spec"] = df["medical_specialty"].str.strip()
    df["text"] = (df["transcription"].fillna("")
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    df = df[df["text"] != ""].copy()
    print(f"usable rows: {len(df)} | unique notes: {df['text'].nunique()}")

    for mode, path in [("specialty", "data/mt_specialty.csv"),
                       ("strict", "data/mt_strict.csv")]:
        out, stats = build(df, mode)
        out.to_csv(path, index=False)
        print(f"\n[{mode}] -> {path}  ({len(out)} notes, {stats['conflicts']} conflicts dropped)")
        for lab, n in out["label"].value_counts().items():
            print(f"  {lab:<18} {n:>5} ({100*n/len(out):4.1f}%)")


if __name__ == "__main__":
    main()