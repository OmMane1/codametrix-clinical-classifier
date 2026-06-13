"""Create the team's FROZEN train/validation split — the shared yardstick.

Everyone trains on `data/splits/train.csv` and reports final numbers on
`data/splits/val.csv` via `python -m clinical_classifier.evaluate`. Same split,
same seed, for all three models -> apples-to-apples comparison.

The split is deterministic (stratified, seed=42), so re-running reproduces it
exactly. Each row gets a stable `id` so model predictions can be joined back to
the truth for scoring.

Usage:  python make_split.py            # from data/mt_specialty.csv, 80/20
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from clinical_classifier import config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/mt_specialty.csv")
    p.add_argument("--val-size", type=float, default=0.20)
    p.add_argument("--out-dir", default="data/splits")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data).reset_index(drop=True)
    df.insert(0, "id", df.index)  # stable id = row position in the source file

    train, val = train_test_split(
        df, test_size=args.val_size, stratify=df["label"],
        random_state=config.RANDOM_STATE,
    )
    train = train.sort_values("id")
    val = val.sort_values("id")

    from pathlib import Path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)

    print(f"source: {args.data}  ({len(df)} notes)")
    print(f"train : {len(train)}  -> {out_dir/'train.csv'}")
    print(f"val   : {len(val)}  -> {out_dir/'val.csv'}")
    print("\nval label distribution (the frozen yardstick):")
    for lab, n in val["label"].value_counts().items():
        print(f"  {lab:<18} {n:>4}")
    print(f"\nseed={config.RANDOM_STATE}. Re-running reproduces this split exactly.")


if __name__ == "__main__":
    main()
