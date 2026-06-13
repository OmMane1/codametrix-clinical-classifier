# MTSamples — cleaned 5-class training data

Cleaned training data for the clinical-note classification task, derived from
**MTSamples** (Kaggle: `tboyle10/medicaltranscriptions`). These are real
clinical *transcriptions*, which match the challenge's "paragraph-length patient
notes" far better than the Kaggle `medical-text` research abstracts.

## Files

| File | Notes | Use |
|---|---|---|
| `data/mt_specialty.csv` | 2,012 | **Recommended.** Cleaner "Other". |
| `data/mt_strict.csv` | 2,316 | Alternative. Broader "Other". |

Both are 2-column: **`text`** (the transcription) and **`label`** (one of
`Cardiology, Neurology, Orthopedics, Gastroenterology, Other`).

Regenerate from raw with: `python clean_mtsamples.py`
(needs `data/raw/mtsamples.csv` — download from the Kaggle dataset; it's
gitignored, not in the repo.)

## How it was cleaned

1. **Text** = the `transcription` column, whitespace-normalized. 33 rows with
   empty transcriptions were dropped.
2. **MTSamples is multi-labeled.** There are only 2,357 unique note texts, but
   2,148 of them are filed under *more than one* specialty (e.g. an echo note
   under both "Cardiovascular / Pulmonary" and "Radiology"). We resolve each
   unique note to ONE label by **priority**:
   - a **target specialty** (Cardiology / Neurology / Orthopedics /
     Gastroenterology) beats the generic "Other" / document-type buckets — so a
     note tagged both "Surgery" and "Orthopedic" becomes **Orthopedics**;
   - if **two different target specialties** conflict → the note is dropped
     (only 41 notes);
   - otherwise → **Other**.
3. This priority step is what makes the data usable — a naive dedupe assigned
   target-class notes to "Surgery"/"Radiology" and collapsed Cardiology to ~26.

## strict vs specialty — the only difference is how "Other" is built

MTSamples mixes real specialties with **document-type** buckets ("Surgery",
"Radiology", "Discharge Summary", "Consult"…) that span every body system.

- **strict** — every non-target label becomes Other (document-type buckets
  included). Bigger, noisier Other.
- **specialty** — document-type buckets are *dropped*; Other = genuine
  non-target specialties only (Urology, Dermatology, OB/GYN…). Cleaner Other.

### Class distribution

| Class | specialty | strict |
|---|---|---|
| Other | 921 (45.8%) | 1,225 (52.9%) |
| Cardiology | 366 (18.2%) | 366 (15.8%) |
| Orthopedics | 320 (15.9%) | 320 (13.8%) |
| Gastroenterology | 220 (10.9%) | 220 (9.5%) |
| Neurology | 185 (9.2%) | 185 (8.0%) |

### Baseline check (TF-IDF word+char + LogisticRegression, 5-fold CV macro-F1)

| Variant | macro-F1 | Orthopedics F1 |
|---|---|---|
| **specialty** | **0.899** | 0.915 |
| strict | 0.861 | 0.893 |

## ⚠️ Open question for the organizers

`specialty` scores higher in cross-validation partly because we made "Other"
cleaner than reality. The **hidden test's "Other" may include Surgery/Radiology/
General-Medicine-style notes** that `specialty` never trained on. So
cleaner-in-CV ≠ guaranteed-better-on-the-hidden-test. We've asked the organizers
how "Other" is defined; until then, keep `strict` around as the robust fallback.

Also note: CV macro-F1 (~0.90) is likely optimistic — MTSamples has its own
dictation style; the real test notes may read differently.
