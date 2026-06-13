"""The model: TF-IDF features (word + char n-grams) -> Logistic Regression.

Why these choices for a *strong* baseline:
  * word (1,2)-grams capture clinical phrases ("chest pain", "bowel movement").
  * char_wb (3,5)-grams add robustness to medical morphology / misspellings
    ("cardi-", "-itis", "gastro-") and OOV terms.
  * sublinear_tf + smoothed idf is the standard text-classification setup.
  * LogisticRegression with class_weight="balanced" handles class imbalance and
    gives calibrated-ish probabilities (useful if you later threshold "Other").
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from . import config


def _make_clf(clf: str, C: float):
    if clf == "logreg":
        return LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",  # robust for sparse, smallish multiclass (one-vs-rest)
            random_state=config.RANDOM_STATE,
        )
    if clf == "linearsvc":
        return LinearSVC(
            C=C,
            class_weight="balanced",
            dual=True,            # n_features >> n_samples for TF-IDF
            max_iter=5000,
            random_state=config.RANDOM_STATE,
        )
    raise ValueError(f"unknown clf {clf!r}; use 'logreg' or 'linearsvc'")


def build_pipeline(
    *,
    clf: str = "logreg",
    C: float = 10.0,
    word_ngram=(1, 2),
    char_ngram=(3, 5),
    min_df: int = 2,
    max_df: float = 0.9,
    use_char: bool = True,
    max_features=None,
) -> Pipeline:
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b",  # drop bare numbers
        max_features=max_features,
    )

    if use_char:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            min_df=min_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    else:
        features = FeatureUnion([("word", word_vec)])

    return Pipeline([("features", features), ("clf", _make_clf(clf, C))])


# Grid for tuning. Keys are pipeline param paths; keep it small so it runs fast
# on ~1k notes. Expand if you have time budget.
PARAM_GRID = {
    "clf__C": [1.0, 3.0, 10.0, 30.0],
    "features__word__ngram_range": [(1, 1), (1, 2)],
    "features__word__min_df": [1, 2, 3],
}
