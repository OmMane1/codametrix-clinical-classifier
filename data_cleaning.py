import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download once if needed
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


DEFAULT_COLUMNS = {
    0: "conditions",
    1: "full_text",
}


def normalize_whitespace(text: str) -> str:
    """Convert missing values to empty strings and normalize whitespace."""

    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_for_encoder(text: str) -> str:
    """
    Minimal cleaning for clinical encoders such as BioClinicalBERT.

    Keep casing, digits, punctuation, stopwords, and negations because they can
    carry clinical meaning: ages, dosages, lab values, anatomy levels, and
    phrases such as "no chest pain".
    """

    return normalize_whitespace(text)


def remove_stopwords(text: str) -> str:
    """
    Notebook's Cleaning(text) function:
    - split text into tokens
    - remove English stopwords
    - join tokens back into string
    """
    tokens = []

    for token in text.split():
        if token not in stop_words:
            tokens.append(token)

    return " ".join(tokens)


def clean_for_classical_model(
    text: str,
    *,
    remove_digits: bool = False,
    remove_stop_words: bool = False,
) -> str:
    """
    Conservative cleaning for TF-IDF / classical models.

    Defaults preserve digits and stopwords because both can be predictive in
    clinical text. Use the keyword arguments only for experiments where CV shows
    they help.
    """

    text = normalize_whitespace(text).lower()

    if remove_digits:
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\(\s*\)", " ", text)

    text = re.sub(r"\.+", ".", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"\s+", " ", text)

    if remove_stop_words:
        text = remove_stopwords(text)

    return text.strip()


def clean_medical_text(text: str) -> str:
    """
    Backward-compatible alias for older notebook code.

    New code should prefer either:
    - clean_for_encoder for transformer models
    - clean_for_classical_model for TF-IDF / linear models
    """

    return clean_for_classical_model(text)


def prepare_medical_dataframe(
    path: str,
    *,
    drop_conditions: set[int] | None = None,
    label_offset: int = 1,
) -> pd.DataFrame:
    """
    Load the medical text dataset and create model-ready text columns.

    - read train.dat
    - rename columns
    - create essay_id
    - create zero-indexed label
    - create encoder_text for clinical encoders
    - create clean_text for classical models

    By default, no conditions are dropped. The old notebook removed condition 5,
    but that is unsafe for a five-class challenge unless condition 5 is proven to
    be out of scope or invalid.
    """

    df = pd.read_csv(path, sep="\t", header=None)

    df.rename(columns=DEFAULT_COLUMNS, inplace=True)
    df = df.dropna(subset=["conditions", "full_text"]).copy()
    df["conditions"] = df["conditions"].astype(int)

    df["essay_id"] = df.index.map(lambda x: f"000{x}")

    if drop_conditions:
        df = df[~df["conditions"].isin(drop_conditions)].reset_index(drop=True)

    df["label"] = df["conditions"] - label_offset

    df["raw_text"] = df["full_text"].astype(str)
    df["encoder_text"] = df["full_text"].apply(clean_for_encoder)
    df["clean_text"] = df["full_text"].apply(clean_for_classical_model)
    df["text_length"] = df["encoder_text"].str.len()
    df["token_count"] = df["encoder_text"].str.split().str.len()

    return df


def summarize_medical_dataframe(df: pd.DataFrame) -> dict:
    """Return quick checks to decide whether any label should be dropped."""

    return {
        "rows": len(df),
        "conditions": df["conditions"].value_counts().sort_index().to_dict(),
        "labels": df["label"].value_counts().sort_index().to_dict(),
        "empty_encoder_text": int((df["encoder_text"] == "").sum()),
        "duplicate_text_rows": int(df.duplicated("encoder_text").sum()),
    }
