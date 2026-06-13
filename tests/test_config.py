"""Tests for the typed config and its YAML round-trip."""

import pytest

from encoder.config import DEFAULT_LABEL_NAMES, EncoderConfig


def test_default_config_has_five_consistent_labels():
    cfg = EncoderConfig()
    assert cfg.num_labels == 5
    assert len(cfg.label_names) == 5
    assert cfg.label_names == DEFAULT_LABEL_NAMES


def test_config_rejects_label_count_mismatch():
    with pytest.raises(ValueError, match="num_labels"):
        EncoderConfig(num_labels=4)  # but 5 label names -> inconsistent


def test_id_label_maps_are_inverses():
    cfg = EncoderConfig()
    for i, name in cfg.id2label.items():
        assert cfg.label2id[name] == i


def test_yaml_round_trip(tmp_path):
    cfg = EncoderConfig(model_name="some/model", max_length=128, run_name="exp1")
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)

    loaded = EncoderConfig.from_yaml(path)

    assert loaded.model_name == "some/model"
    assert loaded.max_length == 128
    assert loaded.run_name == "exp1"


def test_fold_run_name_format():
    cfg = EncoderConfig(run_name="pubmedbert")
    assert cfg.fold_run_name(3) == "pubmedbert_fold3"
