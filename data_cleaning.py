import argparse
from pathlib import Path
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

DEFAULT_KAGGLE_DATASET = "chaitanyakck/medical-text"

HACKATHON_CLASS_TO_LABEL = {
    "cardiology": 0,
    "neurology": 1,
    "orthopedics": 2,
    "gastroenterology": 3,
    "other": 4,
}

SOURCE_CONDITION_TO_HACKATHON_CLASS = {
    1: "other",  # Neoplasms
    2: "gastroenterology",  # Digestive system diseases
    3: "neurology",  # Nervous system diseases
    4: "cardiology",  # Cardiovascular diseases
    5: "other",  # General pathological conditions
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
    df["hackathon_classification"] = df["conditions"].map(
        SOURCE_CONDITION_TO_HACKATHON_CLASS
    )

    if df["hackathon_classification"].isna().any():
        unknown_conditions = sorted(
            df.loc[df["hackathon_classification"].isna(), "conditions"].unique()
        )
        raise ValueError(
            "Found source conditions without a hackathon mapping: "
            f"{unknown_conditions}"
        )

    df["hackathon_label"] = df["hackathon_classification"].map(
        HACKATHON_CLASS_TO_LABEL
    )

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
        "hackathon_classification": (
            df["hackathon_classification"].value_counts().sort_index().to_dict()
        ),
        "hackathon_labels": df["hackathon_label"].value_counts().sort_index().to_dict(),
        "empty_encoder_text": int((df["encoder_text"] == "").sum()),
        "duplicate_text_rows": int(df.duplicated("encoder_text").sum()),
    }


def download_medical_dataset(dataset: str = DEFAULT_KAGGLE_DATASET) -> Path:
    """Download the Kaggle dataset with kagglehub and return its local path."""

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for automatic download. "
            "Install it with: pip install kagglehub"
        ) from exc

    return Path(kagglehub.dataset_download(dataset))


def find_train_file(dataset_dir: str | Path, filename: str = "train.dat") -> Path:
    """Find train.dat inside the downloaded dataset directory."""

    dataset_dir = Path(dataset_dir)
    direct_path = dataset_dir / filename

    if direct_path.exists():
        return direct_path

    matches = sorted(dataset_dir.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} inside {dataset_dir}. "
            "Pass the file directly with --input if it has a different name."
        )

    return matches[0]


def parse_drop_conditions(values: list[str] | None) -> set[int] | None:
    """Parse optional CLI label exclusions."""

    if not values:
        return None

    return {int(value) for value in values}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and clean the medical text classification dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to train.dat. If omitted, the Kaggle dataset is downloaded.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_KAGGLE_DATASET,
        help=f"KaggleHub dataset id. Default: {DEFAULT_KAGGLE_DATASET}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clean_medical_text.csv"),
        help="Output CSV path. Default: clean_medical_text.csv",
    )
    parser.add_argument(
        "--drop-condition",
        action="append",
        dest="drop_conditions",
        help="Condition label to drop. Can be repeated. Default: keep all labels.",
    )
    parser.add_argument(
        "--label-offset",
        type=int,
        default=1,
        help="Value subtracted from source labels to create zero-indexed labels.",
    )

    args = parser.parse_args()

    if args.input:
        train_path = args.input
        print(f"Using input file: {train_path}")
    else:
        dataset_dir = download_medical_dataset(args.dataset)
        train_path = find_train_file(dataset_dir)
        print(f"Downloaded dataset to: {dataset_dir}")
        print(f"Using training file: {train_path}")

    df = prepare_medical_dataframe(
        train_path,
        drop_conditions=parse_drop_conditions(args.drop_conditions),
        label_offset=args.label_offset,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote cleaned dataset to: {args.output}")
    print("Summary:")
    print(summarize_medical_dataframe(df))


if __name__ == "__main__":
    main()
