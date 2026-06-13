"""Cross-validated fine-tuning that produces out-of-fold (OOF) probabilities.

Design goals:

* **No fold leakage.** A *fresh* model is built inside every fold via
  :func:`encoder.model.build_model`. (The reference notebook reused one model
  across folds — that inflates scores and is the headline bug we avoid.)
* **OOF probabilities, not just argmax.** For every training row we store the
  softmax probability vector produced while that row was in the validation
  fold. This ``(n_samples, num_labels)`` array is the artefact the TF-IDF
  teammate needs in order to weight an ensemble — averaging hard labels throws
  that information away.
* **4 GB-friendly.** fp16, gradient accumulation and dynamic padding keep peak
  memory low.

Run via the CLI: ``python -m encoder train --config configs/pubmedbert.yaml``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import EncoderConfig
from .data import (
    add_stratified_folds,
    compute_class_weights,
    load_clean_csv,
    load_training_data,
)
from .metrics import (
    classification_scores,
    compute_metrics_for_trainer,
    per_class_f1,
    softmax,
)
from .model import build_model

logger = logging.getLogger(__name__)


def _build_tokenized_dataset(frame: pd.DataFrame, tokenizer, config: EncoderConfig):
    """Tokenise a DataFrame slice into a HF Dataset with input_ids + labels."""
    from datasets import Dataset

    ds = Dataset.from_pandas(frame[[config.text_column, config.label_column]], preserve_index=False)

    def _tok(batch):
        enc = tokenizer(
            batch[config.text_column],
            truncation=True,
            max_length=config.max_length,
        )
        enc["labels"] = batch[config.label_column]
        return enc

    return ds.map(_tok, batched=True, remove_columns=[config.text_column, config.label_column])


def train_cv(
    config: EncoderConfig,
    cleaner: Optional[Callable[[str], str]] = None,
) -> dict:
    """Run cross-validated training and write OOF probabilities + models.

    Returns a results dict with overall OOF metrics and per-fold metrics.
    """
    # Imports are local so that unit tests importing this module do not require
    # torch/transformers to be installed.
    import torch
    from torch import nn
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    class WeightedTrainer(Trainer):
        """Trainer with an optional class-weighted cross-entropy loss.

        ``class_weights`` is a 1-D tensor of length ``num_labels`` placed on the
        model's device at loss time. ``**kwargs`` absorbs version differences in
        the ``compute_loss`` signature (e.g. ``num_items_in_batch``).
        """

        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = (
                self._class_weights.to(logits.device)
                if self._class_weights is not None
                else None
            )
            loss = nn.functional.cross_entropy(logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    if config.data_format == "csv":
        df = load_clean_csv(
            config.data_path,
            text_column=config.text_column,
            label_column=config.label_column,
            num_labels=config.num_labels,
            drop_duplicate_text=config.drop_duplicate_text,
            min_text_chars=config.min_text_chars,
            cleaner=cleaner,
        )
    else:
        df = load_training_data(
            config.data_path,
            text_column=config.text_column,
            label_column=config.label_column,
            cleaner=cleaner,
        )
    df = add_stratified_folds(
        df, label_column=config.label_column, n_folds=config.n_folds, seed=config.seed
    )
    class_counts = df[config.label_column].value_counts().sort_index()
    logger.info("Loaded %d rows; class counts:\n%s", len(df), class_counts)
    absent = [i for i in range(config.num_labels) if i not in class_counts.index]
    if absent:
        logger.warning(
            "No training rows for class(es) %s (%s). The 5-way head is kept for "
            "the hidden test, but these classes cannot be learned from this data.",
            absent,
            [config.label_names[i] for i in absent],
        )

    oof_dir = Path(config.oof_dir)
    oof_dir.mkdir(parents=True, exist_ok=True)
    model_root = Path(config.output_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    collator = DataCollatorWithPadding(tokenizer)

    n = len(df)
    oof_probs = np.full((n, config.num_labels), np.nan, dtype=np.float32)
    y_true = df[config.label_column].to_numpy()

    folds = [0] if config.single_fold else list(range(config.n_folds))
    fold_metrics = {}

    for fold in folds:
        logger.info("==== Fold %d/%d ====", fold, config.n_folds - 1)
        train_df = df[df["fold"] != fold]
        valid_df = df[df["fold"] == fold]
        valid_pos = np.where(df["fold"].to_numpy() == fold)[0]

        train_ds = _build_tokenized_dataset(train_df, tokenizer, config)
        valid_ds = _build_tokenized_dataset(valid_df, tokenizer, config)

        model = build_model(config)  # FRESH model every fold — no leakage.

        args = TrainingArguments(
            output_dir=str(model_root / config.fold_run_name(fold)),
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.lr_scheduler_type,
            fp16=config.fp16,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            save_strategy="steps",
            save_steps=config.eval_steps,
            logging_steps=config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model=config.primary_metric,
            greater_is_better=True,
            save_total_limit=config.save_total_limit,
            report_to="none",
            seed=config.seed,
        )

        class_weights = None
        if config.class_weighted_loss:
            weights = compute_class_weights(
                train_df[config.label_column].to_numpy(), config.num_labels
            )
            class_weights = torch.tensor(weights, dtype=torch.float)
            logger.info("Fold %d class weights: %s", fold, weights.round(3).tolist())

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics_for_trainer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            class_weights=class_weights,
        )

        trainer.train()

        logits = trainer.predict(valid_ds).predictions
        probs = softmax(logits)
        oof_probs[valid_pos] = probs

        fold_pred = probs.argmax(axis=1)
        scores = classification_scores(valid_df[config.label_column].to_numpy(), fold_pred)
        fold_metrics[f"fold_{fold}"] = scores
        logger.info("Fold %d scores: %s", fold, scores)

        # Persist the best model for this fold for later ensembling/inference.
        best_dir = model_root / config.fold_run_name(fold)
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))

    # ---- Aggregate OOF results --------------------------------------------
    results = {"per_fold": fold_metrics}
    if not config.single_fold:
        scored_mask = ~np.isnan(oof_probs).any(axis=1)
        oof_pred = oof_probs[scored_mask].argmax(axis=1)
        overall = classification_scores(y_true[scored_mask], oof_pred)
        overall["per_class_f1"] = per_class_f1(
            y_true[scored_mask], oof_pred, config.label_names
        )
        results["overall_oof"] = overall
        logger.info("Overall OOF: %s", overall)

        np.save(oof_dir / f"{config.run_name}_oof_probs.npy", oof_probs)
        np.save(oof_dir / f"{config.run_name}_oof_labels.npy", y_true)

    with open(oof_dir / f"{config.run_name}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    config.to_yaml(model_root / f"{config.run_name}_config.yaml")

    return results
