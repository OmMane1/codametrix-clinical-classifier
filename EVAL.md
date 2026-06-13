# Team evaluation protocol — read before reporting numbers

So all three models (TF-IDF baseline, clinical encoder, transformer) are
comparable, we score everything **the same way on the same frozen split**.

## The frozen split

`make_split.py` carves `data/mt_specialty.csv` into a fixed 80/20 stratified
split (seed=42):

| File | Rows | Use |
|---|---|---|
| `data/splits/train.csv` | 1,609 | **Train / tune on this only.** |
| `data/splits/val.csv` | 403 | **Never train on this.** Final scoring only. |

Columns: `id, text, label`. The split is deterministic — `python make_split.py`
reproduces it exactly.

> ⚠️ **Do not let val notes leak into training.** If you build a different
> "Other" (e.g. the strict variant), rebuild it from `train.csv` only — don't
> pull notes that live in `val.csv`.

## How to report your model's score

1. Train on `data/splits/train.csv`.
2. Predict on `data/splits/val.csv`, writing a CSV with columns **`id`** (matching
   val.csv) and **`prediction`** (one of `Cardiology, Neurology, Orthopedics,
   Gastroenterology, Other`).
3. Score it:
   ```bash
   python -m clinical_classifier.evaluate --pred your_val_preds.csv --truth data/splits/val.csv
   ```
   Or in code:
   ```python
   from clinical_classifier.evaluate import evaluate
   evaluate(y_true, y_pred)   # prints macro-F1, per-class F1, confusion matrix
   ```

**Headline metric = macro-F1** (every class weighted equally, so the 46% "Other"
bucket can't hide weak classes). Report macro-F1 + the per-class table.

## Current leaderboard (frozen val)

| Model | macro-F1 | accuracy | notes |
|---|---|---|---|
| TF-IDF (word+char) + LogReg | **0.903** | 0.913 | `models/tfidf_baseline.joblib` |
| _clinical encoder_ | _tbd_ | | teammate |
| _transformer_ | _tbd_ | | teammate |

Baseline error pattern: weakest recall on the small classes — Neurology (0.81)
and Gastroenterology (0.82), mostly leaking into Other. Worth knowing for
ensembling.

## Final submission

`models/tfidf_submission.joblib` is the baseline refit on **all** of
`mt_specialty.csv` (more data = stronger final model). When the hidden test
arrives:
```bash
python -m clinical_classifier.predict --model models/tfidf_submission.joblib \
    --data <hidden_test.csv> --out submission.csv
```
(Add `--proba` to emit per-class probabilities — needed for ensembling.)
