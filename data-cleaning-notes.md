# Data Cleaning Notes

## Goal

The cleaning pipeline is designed to support two modeling tracks without losing useful clinical information:

- A clinical encoder track, such as BioClinicalBERT or BiomedBERT.
- A classical text model track, such as TF-IDF with logistic regression or linear SVM.

Clinical notes contain useful signal in details that generic cleaning often removes: numbers, punctuation, abbreviations, negation words, dosages, ages, lab values, anatomy levels, and procedure references. For that reason, the pipeline keeps multiple text columns instead of forcing every model to use one aggressively cleaned version.

## Input Format

`prepare_medical_dataframe(path)` expects the source file to be tab-separated with no header:

| Column | Meaning |
|---|---|
| `0` | Original condition/class label |
| `1` | Full clinical text |

The loader renames these columns to:

| New column | Meaning |
|---|---|
| `conditions` | Original label from the dataset |
| `full_text` | Original note text from the dataset |

Rows with missing `conditions` or missing `full_text` are dropped because they cannot be used for supervised text classification.

## Output Columns

The cleaning script creates these model-ready columns:

| Column | Purpose |
|---|---|
| `essay_id` | Stable row identifier based on the original row index |
| `label` | Zero-indexed label created from `conditions - label_offset` |
| `hackathon_classification` | Strictly mapped hackathon class name |
| `hackathon_label` | Numeric hackathon label using the challenge class order |
| `raw_text` | Original text converted to string |
| `encoder_text` | Lightly normalized text for clinical encoders |
| `clean_text` | Conservative lowercase text for classical models |
| `text_length` | Character length of `encoder_text` |
| `token_count` | Whitespace token count of `encoder_text` |

## Hackathon Class Mapping

The Kaggle dataset classes do not exactly match the hackathon classes. The
pipeline therefore creates a strict rule-based mapping into the hackathon label
space:

| Source `conditions` | Source meaning | `hackathon_classification` | `hackathon_label` |
|---:|---|---|---:|
| `1` | Neoplasms | `other` | `4` |
| `2` | Digestive system diseases | `gastroenterology` | `3` |
| `3` | Nervous system diseases | `neurology` | `1` |
| `4` | Cardiovascular diseases | `cardiology` | `0` |
| `5` | General pathological conditions | `other` | `4` |

The challenge class order is:

| `hackathon_label` | Class |
|---:|---|
| `0` | Cardiology |
| `1` | Neurology |
| `2` | Orthopedics |
| `3` | Gastroenterology |
| `4` | Other |

There are currently no source rows mapped to `orthopedics`, so that class is
present in the label space but empty in this cleaned dataset. Models trained
only on this data should not be expected to learn orthopedics without additional
orthopedic examples.

## Encoder Cleaning

Function: `clean_for_encoder`

This is intentionally minimal:

1. Convert missing values to an empty string.
2. Convert the value to a string.
3. Normalize repeated whitespace.
4. Strip leading and trailing whitespace.

It does not lowercase, remove digits, remove punctuation, or remove stopwords.

This is the right default for BioClinicalBERT/BiomedBERT because pretrained encoders already have their own tokenizers and were trained on natural biomedical or clinical text. Over-cleaning can make the input less similar to the text the model saw during pretraining.

Examples of information we keep:

| Text feature | Why it may matter |
|---|---|
| Digits | Ages, dosages, lab values, vertebral levels, dates |
| Stopwords | Negation and clinical phrasing, such as "no chest pain" |
| Punctuation | Abbreviations, measurements, note structure |
| Casing | Some abbreviations or section cues may carry signal |

For the clinical encoder track, use:

```python
texts = df["encoder_text"]
labels = df["label"]
```

## Classical Model Cleaning

Function: `clean_for_classical_model`

This cleaning is still conservative:

1. Normalize whitespace.
2. Lowercase text.
3. Optionally remove digits.
4. Normalize repeated periods and commas.
5. Optionally remove stopwords.
6. Strip leading and trailing whitespace.

By default, the function keeps digits and stopwords:

```python
clean_for_classical_model(text)
```

Optional experiments:

```python
clean_for_classical_model(text, remove_digits=True)
clean_for_classical_model(text, remove_stop_words=True)
clean_for_classical_model(text, remove_digits=True, remove_stop_words=True)
```

These options should only be used if cross-validation shows they improve performance. In clinical text, removing digits or stopwords can hurt because they may contain specialty-specific signal.

For the TF-IDF track, use:

```python
texts = df["clean_text"]
labels = df["label"]
```

## Why Condition 5 Is No Longer Dropped by Default

The original notebook removed rows where:

```python
conditions == 5
```

That behavior is now optional through:

```python
prepare_medical_dataframe(path, drop_conditions={5})
```

The default is to keep every condition.

Reasoning:

- The hackathon task is described as a five-class classification problem.
- If labels are `1, 2, 3, 4, 5`, then condition `5` is likely a valid class.
- Dropping condition `5` would silently turn a five-class problem into a four-class problem.
- If condition `5` is the "Other" class, dropping it would remove one of the hardest and most important categories.

Only drop condition `5` if the dataset documentation or challenge organizers explicitly say it is invalid, unlabeled, corrupted, or outside the scoring task.

## Label Handling

The script creates zero-indexed labels:

```python
df["label"] = df["conditions"] - label_offset
```

The default `label_offset` is `1`, so source labels:

```text
1, 2, 3, 4, 5
```

become:

```text
0, 1, 2, 3, 4
```

This format is convenient for PyTorch, Hugging Face Transformers, scikit-learn, and NumPy probability arrays.

## Sanity Checks

Use `summarize_medical_dataframe(df)` after loading:

```python
from data_cleaning import prepare_medical_dataframe, summarize_medical_dataframe

df = prepare_medical_dataframe("train.dat")
print(summarize_medical_dataframe(df))
```

This returns:

| Field | Meaning |
|---|---|
| `rows` | Number of usable rows |
| `conditions` | Class counts using original labels |
| `labels` | Class counts using zero-indexed labels |
| `empty_encoder_text` | Number of rows with empty model input |
| `duplicate_text_rows` | Number of duplicated note texts |

These checks help answer whether any label should be dropped, whether class imbalance is severe, and whether duplicate notes may affect validation.

## Recommended Usage

For the clinical encoder:

```python
df = prepare_medical_dataframe("train.dat")
texts = df["encoder_text"]
labels = df["hackathon_label"]
```

For TF-IDF or linear models:

```python
df = prepare_medical_dataframe("train.dat")
texts = df["clean_text"]
labels = df["hackathon_label"]
```

For experiments that intentionally exclude a condition:

```python
df = prepare_medical_dataframe("train.dat", drop_conditions={5})
```

Do this only when there is a clear dataset reason.

## Summary

The central decision is to preserve clinical information by default. The pipeline keeps a transformer-friendly text field, a classical-model text field, and enough summary metadata to inspect the dataset before modeling. This makes the cleaning process safer for a small-data hackathon, where throwing away the wrong token or label can hurt more than it helps.
