# Encoder pipeline

Fine-tunes a biomedical transformer (PubMedBERT by default) on the Kaggle
**Medical Abstracts** corpus (`chaitanyakck/medical-text`) and predicts the
organiser's hidden `Case N` test file.

## Why PubMedBERT?

The corpus is PubMed-style abstracts, and PubMedBERT was pretrained from
scratch on PubMed — the tightest domain match (tops the BLURB benchmark).
BioClinicalBERT (MIMIC clinical notes) is the wrong register; DeBERTa-v3-xsmall
is the fast/light fallback for a 4 GB GPU.

## Labels (all five — class 5 is NOT dropped)

| raw | index | name                              |
|-----|-------|-----------------------------------|
| 1   | 0     | neoplasms                         |
| 2   | 1     | digestive_system_diseases         |
| 3   | 2     | nervous_system_diseases           |
| 4   | 3     | cardiovascular_diseases           |
| 5   | 4     | general_pathological_conditions   |

## Layout (`src/encoder/`)

```
config.py   typed config (+ YAML)            data.py    loaders + Case-N parser + folds
model.py    factory w/ 4GB-safe freezing     metrics.py macro-F1 (primary), acc, weighted-F1
train.py    fold-correct CV -> OOF probs      predict.py fold ensemble -> submission.csv
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

Put `train.dat` in `data/raw/`.

## Usage

```powershell
# Fast end-to-end smoke check on the light model, one fold:
python -m encoder train --config configs/deberta_xsmall.yaml --single-fold

# Full CV with the quality model (writes artifacts/oof/pubmedbert_oof_probs.npy):
python -m encoder train --config configs/pubmedbert.yaml

# Predict on the hidden test file (ensembles all saved folds):
python -m encoder predict --config configs/pubmedbert.yaml --test-path data/raw/test_cases.txt
```

Add `--use-cleaner` to apply the teammate's `data_cleaning.clean_medical_text`
to both train and test (keep it consistent across both, or neither).

## Deliverable to the team

`artifacts/oof/<run>_oof_probs.npy` — `(n_train, 5)` out-of-fold probabilities,
aligned with `<run>_oof_labels.npy`. The TF-IDF owner ensembles against these.
`artifacts/submissions/<run>_test_probs.npy` is the matching test matrix.

## Design notes

* **Fresh model per fold** in `train_cv` — no cross-fold weight leakage (the
  bug present in the reference notebook, which reused one model across folds).
* **OOF probabilities**, not argmax, so the ensemble keeps full information.
* `model.py` freezing is architecture-agnostic (BERT/DeBERTa) and degrades
  gracefully with a warning.
* `data`/`metrics`/`config` have no torch dependency, so `pytest` runs without
  a GPU or the transformers stack.

## Tests

```powershell
pytest -q          # data parsing, fold coverage, label mapping, metrics, config
```
