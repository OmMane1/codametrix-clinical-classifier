"""Clinical-abstract encoder package.

A small, well-tested pipeline for fine-tuning a biomedical transformer
(PubMedBERT by default) on the Medical Abstracts corpus and predicting the
organiser's hidden ``Case N`` test file.

Public API:
    EncoderConfig            - typed run configuration
    load_training_data       - parse the tab-separated train corpus
    parse_case_file          - parse the hidden-test "Case N" file
    add_stratified_folds     - stratified K-fold assignment
    build_model              - model factory with layer freezing
    train_cv                 - cross-validated training -> OOF probabilities
    predict_test             - fold-ensemble inference -> submission
"""

from .config import DEFAULT_LABEL_NAMES, EncoderConfig
from .data import add_stratified_folds, load_training_data, parse_case_file
from .metrics import classification_scores, per_class_f1, softmax

__all__ = [
    "EncoderConfig",
    "DEFAULT_LABEL_NAMES",
    "load_training_data",
    "parse_case_file",
    "add_stratified_folds",
    "classification_scores",
    "per_class_f1",
    "softmax",
    "build_model",
    "train_cv",
    "predict_test",
]


# ``build_model``/``train_cv``/``predict_test`` pull in torch/transformers, so
# import them lazily — the data/metrics utilities stay usable (and testable)
# without the heavy ML stack installed.
def __getattr__(name):  # PEP 562
    if name == "build_model":
        from .model import build_model

        return build_model
    if name == "train_cv":
        from .train import train_cv

        return train_cv
    if name == "predict_test":
        from .predict import predict_test

        return predict_test
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
