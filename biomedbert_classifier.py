import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_cleaning import HACKATHON_CLASS_TO_LABEL, prepare_medical_dataframe


MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
ID_TO_CLASS = {label: name for name, label in HACKATHON_CLASS_TO_LABEL.items()}
CLASS_NAMES = [ID_TO_CLASS[index] for index in sorted(ID_TO_CLASS)]
DISPLAY_CLASS_NAMES = {
    "cardiology": "Cardiology",
    "neurology": "Neurology",
    "orthopedics": "Orthopedics",
    "gastroenterology": "Gastroenterology",
    "other": "Other",
}
ID_TO_DISPLAY_CLASS = {
    label: DISPLAY_CLASS_NAMES[name] for label, name in ID_TO_CLASS.items()
}
DISPLAY_CLASS_ORDER = [ID_TO_DISPLAY_CLASS[index] for index in sorted(ID_TO_CLASS)]


class ClinicalTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int,
        truncation_strategy: str = "head",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def __len__(self) -> int:
        return len(self.texts)

    def encode_head_tail(self, text: str) -> dict[str, torch.Tensor]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        token_budget = self.max_length - special_tokens

        if len(token_ids) > token_budget:
            head_budget = token_budget // 2
            tail_budget = token_budget - head_budget
            token_ids = token_ids[:head_budget] + token_ids[-tail_budget:]

        token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        attention_mask = [1] * len(token_ids)

        pad_length = self.max_length - len(token_ids)
        if pad_length > 0:
            token_ids += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length

        item = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if "token_type_ids" in self.tokenizer.model_input_names:
            item["token_type_ids"] = torch.zeros(self.max_length, dtype=torch.long)

        return item

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.truncation_strategy == "head_tail":
            item = self.encode_head_tail(self.texts[index])
        else:
            encoded = self.tokenizer(
                self.texts[index],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}

        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def load_training_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = prepare_medical_dataframe(path)

    if {"text", "label"}.issubset(df.columns):
        df = df.rename(
            columns={
                "text": "encoder_text",
                "label": "hackathon_classification",
            }
        )
        df["hackathon_classification"] = (
            df["hackathon_classification"].astype(str).str.strip().str.lower()
        )
        df["hackathon_label"] = df["hackathon_classification"].map(
            HACKATHON_CLASS_TO_LABEL
        )
    else:
        required_columns = {
            "encoder_text",
            "hackathon_label",
            "hackathon_classification",
        }
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Input data is missing required columns: {sorted(missing_columns)}. "
                "Expected either MTSamples columns ['text', 'label'] or the cleaned "
                "columns from data_cleaning.py."
            )

        df["hackathon_classification"] = (
            df["hackathon_classification"].astype(str).str.strip().str.lower()
        )

    df = df.dropna(subset=["encoder_text", "hackathon_label"]).copy()
    df["encoder_text"] = df["encoder_text"].astype(str)
    df["hackathon_label"] = df["hackathon_label"].astype(int)

    if "essay_id" not in df.columns and "id" in df.columns:
        df["essay_id"] = df["id"].astype(str)

    if "essay_id" not in df.columns:
        df["essay_id"] = [f"mt_{index:05d}" for index in range(len(df))]

    return df


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            average="weighted",
            zero_division=0,
        ),
    }


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    if args.warmup_ratio > 0:
        print(
            "Using warmup_ratio. If your Transformers version warns that this is "
            "deprecated, training can still continue."
        )

    common_args = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": 25,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
        "fp16": args.fp16,
    }

    try:
        return TrainingArguments(evaluation_strategy="epoch", **common_args)
    except TypeError:
        return TrainingArguments(eval_strategy="epoch", **common_args)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)

        loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=class_weights,
        )

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: pd.Series, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = labels.value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)

    if (counts == 0).any():
        missing = [ID_TO_CLASS[index] for index, count in counts.items() if count == 0]
        raise ValueError(f"Cannot compute class weights with missing classes: {missing}")

    if mode == "balanced":
        weights = len(labels) / (len(CLASS_NAMES) * counts)
    elif mode == "sqrt":
        weights = np.sqrt(len(labels) / (len(CLASS_NAMES) * counts))
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")

    weights = weights / weights.mean()
    return torch.tensor(weights.to_numpy(dtype=np.float32), dtype=torch.float32)


def warn_about_missing_classes(labels: pd.Series) -> None:
    present_labels = set(labels.astype(int).unique())
    missing_labels = sorted(set(range(len(CLASS_NAMES))) - present_labels)

    if not missing_labels:
        return

    missing_names = [ID_TO_CLASS[label] for label in missing_labels]
    print(
        "Warning: no training rows found for these hackathon classes: "
        f"{missing_names}. The model will still output 5 classes, but it cannot "
        "learn those classes from this dataset alone."
    )


def print_class_counts(df: pd.DataFrame, title: str = "Training data class counts") -> None:
    counts = (
        df["hackathon_classification"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    print(f"{title}:")
    for class_name, count in counts.items():
        print(f"  {class_name:<18} {count:>5}")


def save_label_mapping(output_dir: Path) -> None:
    mapping = {
        "model_name": MODEL_NAME,
        "class_to_label": HACKATHON_CLASS_TO_LABEL,
        "id_to_class": {str(key): value for key, value in ID_TO_CLASS.items()},
        "id_to_display_class": {
            str(key): value for key, value in ID_TO_DISPLAY_CLASS.items()
        },
    }

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BiomedBERT for the hackathon clinical classifier."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mt_specialty.csv"),
        help=(
            "Dataset for optional random split experiments. Ignored unless "
            "--allow-random-split is set."
        ),
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/splits/train.csv"),
        help="Frozen training split CSV. If provided, --val-file is also required.",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/splits/val.csv"),
        help="Frozen validation split CSV. If provided, --train-file is also required.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/biomedbert_classifier"),
        help="Directory for checkpoints and model artifacts.",
    )
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "sqrt", "balanced"],
        default="none",
        help=(
            "Optional weighted cross-entropy. Try 'sqrt' first for imbalanced "
            "small clinical classes."
        ),
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=["head", "head_tail"],
        default="head",
        help=(
            "Head keeps the first max-length tokens. Head_tail keeps the start "
            "and end of long notes."
        ),
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument(
        "--tune-size",
        type=float,
        default=0.15,
        help=(
            "Fraction of the frozen train split used for epoch evaluation and "
            "early stopping. The frozen val split is still final scoring only."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-random-split",
        action="store_true",
        help=(
            "Use --input with a random stratified split. Do not use this for "
            "reporting EVAL.md leaderboard numbers."
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 mixed precision. Enable this on a compatible CUDA GPU.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.allow_random_split:
        df = load_training_dataframe(args.input)
        model_train_df, eval_df = train_test_split(
            df,
            test_size=args.validation_size,
            random_state=args.seed,
            stratify=df["hackathon_label"],
        )
        final_val_df = eval_df
        print(
            "Using random split. Do not report these numbers as EVAL.md frozen "
            "validation results."
        )
    else:
        if bool(args.train_file) != bool(args.val_file):
            raise ValueError("--train-file and --val-file must be provided together.")
        frozen_train_df = load_training_dataframe(args.train_file)
        final_val_df = load_training_dataframe(args.val_file)
        print(f"Using frozen split: train={args.train_file}, val={args.val_file}")
        model_train_df, eval_df = train_test_split(
            frozen_train_df,
            test_size=args.tune_size,
            random_state=args.seed,
            stratify=frozen_train_df["hackathon_label"],
        )
        print(
            "Using an internal tune split from train.csv for epoch evaluation; "
            "val.csv is not used until final prediction."
        )

    print_class_counts(model_train_df, "Model training split class counts")
    print_class_counts(eval_df, "Internal eval split class counts")
    print_class_counts(final_val_df, "Frozen validation split class counts")
    warn_about_missing_classes(model_train_df["hackathon_label"])
    class_weights = compute_class_weights(
        model_train_df["hackathon_label"],
        args.class_weighting,
    )

    if class_weights is not None:
        print(f"Using {args.class_weighting} class weights: {class_weights.tolist()}")
    print(f"Using truncation strategy: {args.truncation_strategy}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        id2label=ID_TO_CLASS,
        label2id=HACKATHON_CLASS_TO_LABEL,
    )

    train_dataset = ClinicalTextDataset(
        model_train_df["encoder_text"].tolist(),
        model_train_df["hackathon_label"].tolist(),
        tokenizer,
        args.max_length,
        args.truncation_strategy,
    )
    eval_dataset = ClinicalTextDataset(
        eval_df["encoder_text"].tolist(),
        eval_df["hackathon_label"].tolist(),
        tokenizer,
        args.max_length,
        args.truncation_strategy,
    )
    final_val_dataset = ClinicalTextDataset(
        final_val_df["encoder_text"].tolist(),
        final_val_df["hackathon_label"].tolist(),
        tokenizer,
        args.max_length,
        args.truncation_strategy,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=build_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    trainer.train()
    metrics = trainer.evaluate()
    predictions = trainer.predict(final_val_dataset)
    val_pred_labels = np.argmax(predictions.predictions, axis=1)

    report = classification_report(
        final_val_df["hackathon_label"],
        val_pred_labels,
        labels=list(range(len(CLASS_NAMES))),
        target_names=DISPLAY_CLASS_ORDER,
        zero_division=0,
        output_dict=True,
    )
    final_val_metrics = {
        "accuracy": accuracy_score(final_val_df["hackathon_label"], val_pred_labels),
        "macro_f1": f1_score(
            final_val_df["hackathon_label"],
            val_pred_labels,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            final_val_df["hackathon_label"],
            val_pred_labels,
            labels=list(range(len(CLASS_NAMES))),
            average="weighted",
            zero_division=0,
        ),
    }

    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    save_label_mapping(args.output_dir)

    np.save(args.output_dir / "validation_logits.npy", predictions.predictions)
    np.save(
        args.output_dir / "validation_labels.npy",
        final_val_df["hackathon_label"].to_numpy(),
    )
    prediction_df = pd.DataFrame(
        {
            "id": final_val_df["essay_id"].astype(str).to_numpy(),
            "prediction": [ID_TO_DISPLAY_CLASS[int(label)] for label in val_pred_labels],
        }
    )
    prediction_df.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    final_val_df[["essay_id", "hackathon_classification", "hackathon_label"]].to_csv(
        args.output_dir / "validation_rows.csv",
        index=False,
    )

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "internal_eval": metrics,
                "frozen_val": final_val_metrics,
                "frozen_val_classification_report": report,
                "protocol": {
                    "train_file": str(args.train_file),
                    "val_file": str(args.val_file),
                    "val_used_for_training_or_early_stopping": args.allow_random_split,
                    "class_weighting": args.class_weighting,
                    "truncation_strategy": args.truncation_strategy,
                },
            },
            file,
            indent=2,
        )

    print("Evaluation metrics:")
    print("Internal eval used during training:")
    print(metrics)
    print("Frozen validation score for EVAL.md reporting:")
    print(final_val_metrics)
    print(
        "Wrote EVAL.md-compatible predictions to: "
        f"{args.output_dir / 'validation_predictions.csv'}"
    )
    print(
        "Score with: python -m clinical_classifier.evaluate "
        f"--pred {args.output_dir / 'validation_predictions.csv'} "
        f"--truth {args.val_file}"
    )
    print(f"Saved model and outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
