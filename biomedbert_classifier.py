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


class ClinicalTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
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

    required_columns = {"encoder_text", "hackathon_label", "hackathon_classification"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input data is missing required columns: {sorted(missing_columns)}. "
            "Run data_cleaning.py first, or pass the raw train.dat file."
        )

    df = df.dropna(subset=["encoder_text", "hackathon_label"]).copy()
    df["encoder_text"] = df["encoder_text"].astype(str)
    df["hackathon_label"] = df["hackathon_label"].astype(int)
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


def save_label_mapping(output_dir: Path) -> None:
    mapping = {
        "model_name": MODEL_NAME,
        "class_to_label": HACKATHON_CLASS_TO_LABEL,
        "id_to_class": {str(key): value for key, value in ID_TO_CLASS.items()},
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
        default=Path("clean_medical_text.csv"),
        help="Clean CSV or raw train.dat path. Default: clean_medical_text.csv",
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
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 mixed precision. Enable this on a compatible CUDA GPU.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_dataframe(args.input)
    warn_about_missing_classes(df["hackathon_label"])

    train_df, val_df = train_test_split(
        df,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=df["hackathon_label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        id2label=ID_TO_CLASS,
        label2id=HACKATHON_CLASS_TO_LABEL,
    )

    train_dataset = ClinicalTextDataset(
        train_df["encoder_text"].tolist(),
        train_df["hackathon_label"].tolist(),
        tokenizer,
        args.max_length,
    )
    val_dataset = ClinicalTextDataset(
        val_df["encoder_text"].tolist(),
        val_df["hackathon_label"].tolist(),
        tokenizer,
        args.max_length,
    )

    trainer = Trainer(
        model=model,
        args=build_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    predictions = trainer.predict(val_dataset)
    val_pred_labels = np.argmax(predictions.predictions, axis=1)

    report = classification_report(
        val_df["hackathon_label"],
        val_pred_labels,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    save_label_mapping(args.output_dir)

    np.save(args.output_dir / "validation_logits.npy", predictions.predictions)
    np.save(args.output_dir / "validation_labels.npy", val_df["hackathon_label"].to_numpy())
    val_df[["essay_id", "hackathon_classification", "hackathon_label"]].to_csv(
        args.output_dir / "validation_rows.csv",
        index=False,
    )

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"eval": metrics, "classification_report": report}, file, indent=2)

    print("Evaluation metrics:")
    print(metrics)
    print(f"Saved model and outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
