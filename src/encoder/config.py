"""Typed configuration for the clinical-abstract encoder.

All hyper-parameters and paths live here so that a run is fully described by a
single object that can be (de)serialised to YAML. This keeps experiments
reproducible and makes the CLI a thin wrapper around a config file.

Defaults are tuned for a 4 GB GPU (e.g. GTX 1650): small batch + gradient
accumulation + fp16 + layer freezing. Override anything via a YAML file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None


# Canonical label set for the Kaggle "medical-text" (Medical Abstracts) corpus.
# The raw .dat files use 1-based labels; we store them 0-indexed internally.
# NOTE: we keep ALL FIVE classes. Class 5 must not be dropped (the hidden test
# spans every class).
DEFAULT_LABEL_NAMES: List[str] = [
    "neoplasms",                      # raw label 1 -> index 0
    "digestive_system_diseases",      # raw label 2 -> index 1
    "nervous_system_diseases",        # raw label 3 -> index 2
    "cardiovascular_diseases",        # raw label 4 -> index 3
    "general_pathological_conditions",  # raw label 5 -> index 4
]


@dataclass
class EncoderConfig:
    """Everything needed to reproduce a training/prediction run."""

    # --- Model ---
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    num_labels: int = 5
    label_names: List[str] = field(default_factory=lambda: list(DEFAULT_LABEL_NAMES))
    max_length: int = 256

    # Regularisation / stabilisation on small-ish data + small GPU.
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    freeze_embeddings: bool = True
    freeze_layers: int = 2  # freeze the lowest N transformer layers

    # --- Optimisation ---
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: float = 3.0
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2  # effective batch = 8 * 2 = 16
    fp16: bool = True
    lr_scheduler_type: str = "linear"

    # --- Cross-validation ---
    n_folds: int = 5
    seed: int = 2024
    # If True, train a single holdout fold (fold 0) only — fast iteration on a
    # slow GPU. Full OOF still requires n_folds runs.
    single_fold: bool = False

    # --- Data ---
    train_path: str = "data/raw/train.dat"
    text_column: str = "text"
    label_column: str = "label"

    # --- Outputs ---
    output_dir: str = "artifacts/models"
    oof_dir: str = "artifacts/oof"
    submission_dir: str = "artifacts/submissions"
    run_name: str = "pubmedbert"

    # --- Eval / logging ---
    primary_metric: str = "macro_f1"
    logging_steps: int = 100
    eval_steps: int = 250
    save_total_limit: int = 1

    def __post_init__(self) -> None:
        if self.num_labels != len(self.label_names):
            raise ValueError(
                f"num_labels ({self.num_labels}) != len(label_names) "
                f"({len(self.label_names)}). Keep them consistent — do not drop "
                "a class silently."
            )

    # --- (de)serialisation -------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "EncoderConfig":
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs.")
        with open(path, "r", encoding="utf-8") as fh:
            data: Dict[str, Any] = yaml.safe_load(fh) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        if yaml is None:
            raise ImportError("PyYAML is required to write YAML configs.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # --- convenience -------------------------------------------------------
    @property
    def id2label(self) -> Dict[int, str]:
        return {i: name for i, name in enumerate(self.label_names)}

    @property
    def label2id(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.label_names)}

    def fold_run_name(self, fold: int) -> str:
        return f"{self.run_name}_fold{fold}"
