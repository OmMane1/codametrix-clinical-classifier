"""Command-line entrypoint: ``python -m encoder <command> [options]``.

Commands
--------
train    Fine-tune with cross-validation and write OOF probabilities.
predict  Run the fold ensemble over a hidden-test ``Case N`` file.

The CLI is intentionally thin — it loads a config (defaults or YAML), applies a
few overrides, and delegates to :mod:`encoder.train` / :mod:`encoder.predict`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from .config import EncoderConfig


def _load_config(args: argparse.Namespace) -> EncoderConfig:
    config = EncoderConfig.from_yaml(args.config) if args.config else EncoderConfig()
    if getattr(args, "model_name", None):
        config.model_name = args.model_name
    if getattr(args, "single_fold", False):
        config.single_fold = True
    return config


def _maybe_cleaner(use_cleaner: bool):
    """Optionally wire in the teammate's text cleaner from data_cleaning.py."""
    if not use_cleaner:
        return None
    try:
        # data_cleaning.py lives at the repo root.
        from data_cleaning import clean_medical_text  # type: ignore

        return clean_medical_text
    except Exception as exc:  # pragma: no cover
        logging.getLogger(__name__).warning(
            "Could not import clean_medical_text (%s); proceeding without cleaning.", exc
        )
        return None


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(prog="encoder", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="Path to a YAML config file.")
    common.add_argument("--model-name", dest="model_name", help="Override model name.")
    common.add_argument(
        "--use-cleaner",
        action="store_true",
        help="Apply data_cleaning.clean_medical_text to text.",
    )

    p_train = sub.add_parser("train", parents=[common], help="Cross-validated training.")
    p_train.add_argument(
        "--single-fold",
        action="store_true",
        help="Train only fold 0 (fast iteration on a slow GPU).",
    )

    p_pred = sub.add_parser("predict", parents=[common], help="Predict on a test file.")
    p_pred.add_argument("--test-path", required=True, help="Path to the Case N test file.")
    p_pred.add_argument(
        "--zero-based",
        action="store_true",
        help="Emit 0-based label ids instead of the default 1-based.",
    )

    args = parser.parse_args(argv)
    config = _load_config(args)
    cleaner = _maybe_cleaner(args.use_cleaner)

    if args.command == "train":
        from .train import train_cv

        results = train_cv(config, cleaner=cleaner)
        print(results.get("overall_oof", results.get("per_fold")))
        return 0

    if args.command == "predict":
        from .predict import predict_test

        submission = predict_test(
            config,
            test_path=args.test_path,
            cleaner=cleaner,
            one_based_output=not args.zero_based,
        )
        print(submission.head())
        print(f"{len(submission)} predictions written.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
