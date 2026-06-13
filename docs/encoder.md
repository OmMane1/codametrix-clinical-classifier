# Encoder pipeline

Fine-tunes a biomedical transformer (PubMedBERT by default) on the cleaned
Medical Abstracts corpus and predicts the organiser's hidden `Case N` test
file, in the **hackathon label space**.

## Why PubMedBERT?

The corpus is PubMed-style abstracts, and PubMedBERT was pretrained from
scratch on PubMed — the tightest domain match (tops the BLURB benchmark).
BioClinicalBERT (MIMIC clinical notes) is the wrong register; DeBERTa-v3-xsmall
is the fast/light fallback for a 4 GB GPU.

## Input data & the CSV contract

The encoder reads `clean_medical_text.csv` (produced by `data_cleaning.py`),
using exactly two columns:

* `encoder_text` — lightly normalised text (casing/digits/punctuation kept),
  the right input for a transformer.
* `hackathon_label` — the 0..4 hackathon class.

**This is the contract between data-cleaning and the encoder.** As long as the
cleaning step emits these two columns in the hackathon label space, the encoder
is dataset-agnostic. The team is switching the *source* dataset from the Kaggle
Medical Abstracts corpus to the **Medical Transcriptions** dataset
(`tboyle10/medicaltranscriptions`, which contains orthopedics cases). The
encoder needs **no change** for this — `data_cleaning.py` maps the new
`medical_specialty` values into `hackathon_label`, regenerates the CSV, and the
encoder consumes it unchanged. Orthopedics stops being empty automatically.

To read a raw tab-separated `train.dat` instead, set `data_format: dat`.

## Hackathon labels

| hackathon_label | name             | source conditions                         |
|----------------:|------------------|-------------------------------------------|
| 0               | cardiology       | cardiovascular diseases                   |
| 1               | neurology        | nervous system diseases                   |
| 2               | orthopedics      | **none — empty in training data**         |
| 3               | gastroenterology | digestive system diseases                 |
| 4               | other            | neoplasms + general pathological conditions |

The source-condition mapping above is for the *Medical Abstracts* corpus, in
which **orthopedics had no rows** (training logged a warning and the class could
not be learned). Switching to the **Medical Transcriptions** dataset adds real
orthopedics examples, so all five classes become learnable. The 5-way head and
the empty-class warning logic stay in place regardless. **Other** remains a
large, heterogeneous class, so the loss is class-weighted by default to protect
macro-F1.

## Layout (`src/encoder/`)

```
config.py   typed config (+ YAML)            data.py    CSV loader, Case-N parser, folds, class weights
model.py    factory w/ 4GB-safe freezing     metrics.py macro-F1 (primary), acc, weighted-F1
train.py    fold-correct CV + weighted loss  predict.py fold ensemble -> submission.csv
cli.py      python -m encoder train|predict
```

## Setup

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .            # makes `python -m encoder` work from anywhere
```

(Without the editable install, prefix commands with `PYTHONPATH=src`.)

Ensure `clean_medical_text.csv` is present at the repo root (or point
`data_path` at it).

## Usage

```powershell
# Fast end-to-end smoke check on the light model, one fold:
python -m encoder train --config configs/deberta_xsmall.yaml --single-fold

# Full CV with the quality model (writes artifacts/oof/pubmedbert_oof_probs.npy):
python -m encoder train --config configs/pubmedbert.yaml

# Predict on the hidden test file (ensembles all saved folds; --use-cleaner
# applies the same clean_for_encoder normalisation used for training):
python -m encoder predict --config configs/pubmedbert.yaml \
    --test-path data/raw/test_cases.txt --use-cleaner
```

## Submission format

`predict` writes `artifacts/submissions/<run>_submission.csv` — the **official
2-column file**: `case_number, prediction`, one row per case. `prediction` is
the class *name* by default (e.g. `cardiology`). Pass `--label-ids` to emit the
numeric `hackathon_label` instead (0-based; add `--one-based` for 1-based).

Alongside it: `<run>_submission_detailed.csv` (case + prediction + per-class
probabilities, for error analysis — not for submission).

## Deliverable to the team

`artifacts/oof/<run>_oof_probs.npy` — `(n_train, 5)` out-of-fold probabilities,
aligned with `<run>_oof_labels.npy`. The TF-IDF owner ensembles against these.
`artifacts/submissions/<run>_test_probs.npy` is the matching test matrix.

## Design notes

* **Fresh model per fold** in `train_cv` — no cross-fold weight leakage (the
  bug present in the reference notebook, which reused one model across folds).
* **OOF probabilities**, not argmax, so the ensemble keeps full information.
* **Class-weighted loss** (inverse frequency over present classes) handles the
  dominant "other" class; absent classes get weight 1.0.
* **Duplicate-text removal** stops the same note leaking across folds.
* `model.py` freezing is architecture-agnostic (BERT/DeBERTa).
* `data`/`metrics`/`config` have no torch dependency, so `pytest` runs without
  a GPU or the transformers stack.

## Tests

```powershell
pytest -q          # CSV loader, Case-N parser, folds, class weights, metrics, config
```
