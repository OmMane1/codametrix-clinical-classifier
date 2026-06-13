"""Model factory.

Builds an ``AutoModelForSequenceClassification`` from a config and applies
memory/stability-oriented layer freezing. Freezing the embeddings and the
lowest few transformer layers both cuts activation memory (key on a 4 GB GPU)
and stabilises fine-tuning on small-to-medium datasets.

The freezing logic is architecture-agnostic: it reaches through
``model.base_model`` so it works for BERT, PubMedBERT, BioClinicalBERT,
DeBERTa-v2/v3, etc., and degrades gracefully (with a warning) if a backbone
exposes an unexpected structure.
"""

from __future__ import annotations

import logging
from typing import List

from transformers import AutoConfig, AutoModelForSequenceClassification

from .config import EncoderConfig

logger = logging.getLogger(__name__)


def _find_encoder_layers(base_model) -> List:
    """Return the list of transformer layer modules, or [] if not found."""
    encoder = getattr(base_model, "encoder", None)
    if encoder is None:
        return []
    # BERT/DeBERTa: encoder.layer ; some models: encoder.layers
    for attr in ("layer", "layers"):
        layers = getattr(encoder, attr, None)
        if layers is not None:
            return list(layers)
    return []


def freeze_parameters(model, freeze_embeddings: bool, freeze_layers: int) -> dict:
    """Freeze embeddings and the lowest ``freeze_layers`` transformer layers.

    Returns a small summary dict (counts) for logging/testing. Mutates the
    model in place by setting ``requires_grad = False`` on selected params.
    """
    base = getattr(model, "base_model", model)
    summary = {"froze_embeddings": False, "froze_layers": 0, "total_frozen_params": 0}

    if freeze_embeddings:
        embeddings = getattr(base, "embeddings", None)
        if embeddings is not None:
            for p in embeddings.parameters():
                p.requires_grad = False
            summary["froze_embeddings"] = True
        else:
            logger.warning("Could not locate embeddings to freeze on %s", type(base).__name__)

    if freeze_layers > 0:
        layers = _find_encoder_layers(base)
        if not layers:
            logger.warning(
                "Could not locate encoder layers to freeze on %s; skipping layer freeze.",
                type(base).__name__,
            )
        else:
            n = min(freeze_layers, len(layers))
            for layer in layers[:n]:
                for p in layer.parameters():
                    p.requires_grad = False
            summary["froze_layers"] = n

    summary["total_frozen_params"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    return summary


def build_model(config: EncoderConfig):
    """Instantiate a fresh classification model from ``config``.

    Always call this *inside* each CV fold so folds never share weights
    (the bug we are deliberately avoiding).
    """
    model_config = AutoConfig.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=config.id2label,
        label2id=config.label2id,
    )
    # Set dropout where the architecture supports these attributes.
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = config.hidden_dropout_prob
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = config.attention_dropout_prob

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, config=model_config
    )

    summary = freeze_parameters(
        model,
        freeze_embeddings=config.freeze_embeddings,
        freeze_layers=config.freeze_layers,
    )
    logger.info("Model built (%s). Freeze summary: %s", config.model_name, summary)
    return model
