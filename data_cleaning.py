import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download once if needed
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

stemmer = SnowballStemmer("english")  # Imported in notebook, but not actually used there


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


def clean_medical_text(text: str) -> str:
    """
    Extracted from notebook's dataPreprocessing(x) function.
    Flow:
    1. lowercase
    2. normalize whitespace
    3. remove digits
    4. normalize repeated periods
    5. normalize repeated commas
    6. remove stopwords
    7. strip leading/trailing whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # 1. lowercase
    text = text.lower()

    # 2. normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # 3. remove digits
    text = re.sub(r"\d+", "", text)

    # 4. notebook had: re.sub('\(d+', '', x)
    # That line likely has a typo and does not do much useful cleaning.
    # If you want to remove things like "(123)", use this:
    text = re.sub(r"\(\d+\)", "", text)

    # 5. normalize repeated periods and commas
    text = re.sub(r"\.+", ".", text)
    text = re.sub(r",+", ",", text)

    # 6. remove stopwords
    text = remove_stopwords(text)

    # 7. strip
    text = text.strip()

    return text


def prepare_medical_dataframe(path: str) -> pd.DataFrame:
    """
    Reconstructs the dataframe setup from the notebook:
    - read train.dat
    - rename columns
    - create essay_id
    - remove condition 5
    - create zero-indexed label
    - clean full_text
    """

    df = pd.read_csv(path, sep="\t", header=None)

    df.rename(
        columns={
            0: "conditions",
            1: "full_text"
        },
        inplace=True
    )

    df["essay_id"] = df.index.map(lambda x: f"000{x}")

    # Notebook removes condition 5
    df = df[df["conditions"] != 5].reset_index(drop=True)

    # Notebook converts labels from 1-based to 0-based
    df["label"] = df["conditions"] - 1

    # Apply cleaning
    df["clean_text"] = df["full_text"].apply(clean_medical_text)

    return df