"""Tests for parsing and fold assignment — the highest-risk logic."""

import pandas as pd
import pytest

from encoder.data import add_stratified_folds, load_training_data, parse_case_file


# --- training corpus loader -------------------------------------------------

def test_load_training_data_maps_labels_to_zero_based(tmp_path):
    f = tmp_path / "train.dat"
    f.write_text("1\tneoplasm abstract text\n5\tgeneral condition text\n", encoding="utf-8")

    df = load_training_data(f)

    assert list(df["raw_label"]) == [1, 5]
    assert list(df["label"]) == [0, 4]  # 1-based -> 0-based
    assert df.loc[0, "text"] == "neoplasm abstract text"


def test_load_training_data_keeps_class_five(tmp_path):
    """Regression guard: class 5 must NOT be dropped (notebook bug)."""
    f = tmp_path / "train.dat"
    f.write_text("\n".join(f"{lbl}\ttext {lbl}" for lbl in [1, 2, 3, 4, 5]), encoding="utf-8")

    df = load_training_data(f)

    assert set(df["raw_label"]) == {1, 2, 3, 4, 5}
    assert set(df["label"]) == {0, 1, 2, 3, 4}


def test_load_training_data_splits_on_first_tab_only(tmp_path):
    f = tmp_path / "train.dat"
    f.write_text("2\ttext with\ta tab inside\n", encoding="utf-8")

    df = load_training_data(f)

    assert df.loc[0, "text"] == "text with\ta tab inside"


def test_load_training_data_rejects_out_of_range_label(tmp_path):
    f = tmp_path / "train.dat"
    f.write_text("7\tsome text\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside 1..5"):
        load_training_data(f)


def test_load_training_data_applies_cleaner(tmp_path):
    f = tmp_path / "train.dat"
    f.write_text("3\tSHOUTING TEXT\n", encoding="utf-8")

    df = load_training_data(f, cleaner=str.lower)

    assert df.loc[0, "text"] == "shouting text"


# --- hidden-test "Case N" parser -------------------------------------------

def test_parse_case_file_basic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text(
        "Case 1\nFirst case body.\n\nCase 2\nSecond case body line one.\nLine two.\n",
        encoding="utf-8",
    )

    df = parse_case_file(f)

    assert list(df["case_id"]) == [1, 2]
    assert df.loc[0, "text"] == "First case body."
    # Multi-line bodies are joined with a space.
    assert df.loc[1, "text"] == "Second case body line one. Line two."


def test_parse_case_file_case_insensitive_header(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("CASE 1\nbody a\ncase 2\nbody b\n", encoding="utf-8")

    df = parse_case_file(f)

    assert list(df["case_id"]) == [1, 2]


def test_parse_case_file_rejects_text_before_first_header(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("stray text\nCase 1\nbody\n", encoding="utf-8")

    with pytest.raises(ValueError, match="before any 'Case N' header"):
        parse_case_file(f)


def test_parse_case_file_rejects_empty_body(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Case 1\n\nCase 2\nbody\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty body"):
        parse_case_file(f)


def test_parse_case_file_rejects_duplicate_ids(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Case 1\nbody a\nCase 1\nbody b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate case ids"):
        parse_case_file(f)


# --- fold assignment --------------------------------------------------------

def test_add_stratified_folds_covers_all_rows_and_is_stratified():
    # 50 rows, 5 classes, 10 each.
    df = pd.DataFrame({"label": [i % 5 for i in range(50)], "text": [f"t{i}" for i in range(50)]})

    out = add_stratified_folds(df, n_folds=5, seed=0)

    assert set(out["fold"]) == {0, 1, 2, 3, 4}
    assert (out["fold"] >= 0).all()
    # Each fold should see every class (balanced design).
    for fold in range(5):
        assert set(out[out["fold"] == fold]["label"]) == {0, 1, 2, 3, 4}


def test_add_stratified_folds_is_deterministic():
    df = pd.DataFrame({"label": [i % 5 for i in range(50)], "text": [f"t{i}" for i in range(50)]})

    a = add_stratified_folds(df, n_folds=5, seed=42)
    b = add_stratified_folds(df, n_folds=5, seed=42)

    assert list(a["fold"]) == list(b["fold"])
