"""Data loading, validation and cleaning policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import (
    DataValidationError,
    check_required_columns,
    clean_dataset,
    dataset_fingerprint,
    load_dataset,
    validate_clustering_input,
)


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------
def test_load_dataset_reads_the_file_and_keeps_expected_columns(synthetic_csv):
    frame = load_dataset(synthetic_csv)
    assert set(config.REQUIRED_COLUMNS).issubset(frame.columns)
    assert len(frame) == 60


def test_missing_required_column_is_reported(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "Age": [30, 40],
            config.INCOME_COLUMN: [20, 30],
        }
    ).to_csv(path, index=False)

    with pytest.raises(DataValidationError) as excinfo:
        load_dataset(path)
    message = str(excinfo.value)
    assert config.SPENDING_SCORE_COLUMN in message


def test_check_required_columns_lists_only_what_is_missing(synthetic_frame):
    assert check_required_columns(synthetic_frame) == []
    trimmed = synthetic_frame.drop(columns=[config.INCOME_COLUMN])
    assert check_required_columns(trimmed) == [config.INCOME_COLUMN]


def test_missing_file_raises_with_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_dataset(tmp_path / "nope.csv")
    message = str(excinfo.value)
    assert "Kaggle" in message
    assert "Mall_Customers.csv" in message


# ---------------------------------------------------------------------------
# Numeric validation and missing / invalid rows
# ---------------------------------------------------------------------------
def test_messy_rows_are_excluded_and_counted(messy_frame):
    dataset = clean_dataset(messy_frame)

    report = dataset.report
    assert report.rows_total == 66
    assert report.rows_kept == 60
    assert report.rows_excluded == 6

    reasons = report.exclusion_reasons
    assert reasons["income_not_numeric_or_missing"] == 2  # NaN and "not a number"
    assert reasons["spending_score_not_numeric_or_missing"] == 1  # NaN only
    assert reasons["income_negative"] == 1
    assert reasons["spending_score_out_of_range"] == 1  # 500
    assert reasons["spending_score_not_whole_number"] == 1  # 42.5
    assert dataset.frame["CustomerID"].tolist() == list(range(1, 61))


def test_non_numeric_predictors_become_nan_and_are_reported(messy_frame):
    dataset = clean_dataset(messy_frame)
    features = dataset.features
    assert features.dtypes.map(lambda d: np.issubdtype(d, np.number)).all()
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert not features.isna().any().any()


def test_cleaning_preserves_source_row_order_and_identifiers(messy_frame):
    dataset = clean_dataset(messy_frame)
    kept = dataset.frame

    assert kept[config.SOURCE_ROW_COLUMN].is_monotonic_increasing
    assert kept[config.SOURCE_ROW_COLUMN].tolist() == list(range(1, 61))
    # Every kept CustomerID still matches its own source row number.
    assert (kept["CustomerID"] == kept[config.SOURCE_ROW_COLUMN]).all()
    # Order of the frame matches order of the source file, not a sort.
    assert kept[config.SOURCE_ROW_COLUMN].tolist() == sorted(
        kept[config.SOURCE_ROW_COLUMN].tolist()
    )


def test_rows_outside_the_documented_limits_are_excluded(synthetic_frame):
    frame = synthetic_frame.copy()
    frame.loc[0, config.SPENDING_SCORE_COLUMN] = 0
    frame.loc[1, config.SPENDING_SCORE_COLUMN] = 101
    frame.loc[2, config.INCOME_COLUMN] = -1

    dataset = clean_dataset(frame)
    assert dataset.report.rows_excluded == 3
    assert dataset.row_count == len(frame) - 3
    assert dataset.report.exclusion_reasons["spending_score_out_of_range"] == 2
    assert dataset.report.exclusion_reasons["income_negative"] == 1


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
def test_customers_sharing_income_and_score_are_all_kept(synthetic_frame):
    duplicated_features = synthetic_frame.copy()
    new_row = len(duplicated_features)
    duplicated_features.loc[new_row] = duplicated_features.iloc[0]
    duplicated_features.loc[new_row, "CustomerID"] = 9999
    duplicated_features.loc[new_row, "Age"] = 44

    dataset = clean_dataset(duplicated_features)
    assert dataset.report.duplicate_feature_pairs == 1
    assert dataset.report.exact_duplicate_rows_removed == 0
    assert dataset.row_count == len(duplicated_features)
    assert 9999 in dataset.frame["CustomerID"].tolist()


def test_exact_duplicate_records_are_removed_and_reported(synthetic_frame):
    with_duplicate = synthetic_frame.copy()
    new_row = len(with_duplicate)
    with_duplicate.loc[new_row] = with_duplicate.iloc[0]

    dataset = clean_dataset(with_duplicate)
    assert dataset.report.exact_duplicate_rows_removed == 1
    assert dataset.row_count == len(synthetic_frame)
    assert dataset.frame["CustomerID"].is_unique


def test_duplicate_removal_can_be_switched_off(synthetic_frame):
    with_duplicate = synthetic_frame.copy()
    new_row = len(with_duplicate)
    with_duplicate.loc[new_row] = with_duplicate.iloc[0]

    dataset = clean_dataset(with_duplicate, drop_exact_duplicates=False)
    assert dataset.report.exact_duplicate_rows_removed == 0
    assert dataset.row_count == len(synthetic_frame) + 1


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def test_only_the_two_predictors_are_exposed_as_features(messy_frame):
    dataset = clean_dataset(messy_frame)
    features = dataset.features

    assert list(features.columns) == list(config.FEATURE_COLUMNS)
    for excluded in config.EXCLUDED_FROM_MODEL:
        assert excluded not in features.columns
        assert excluded in dataset.frame.columns  # kept for traceability only
    assert features.index.is_monotonic_increasing
    assert len(features) == dataset.row_count


# ---------------------------------------------------------------------------
# Clustering preconditions
# ---------------------------------------------------------------------------
def test_too_few_rows_is_refused():
    frame = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            config.INCOME_COLUMN: [10, 20, 30],
            config.SPENDING_SCORE_COLUMN: [1, 2, 3],
        }
    )
    dataset = clean_dataset(frame)
    with pytest.raises(DataValidationError, match="valid row"):
        validate_clustering_input(dataset)


def test_degenerate_constant_features_are_refused():
    rows = 40
    frame = pd.DataFrame(
        {
            "CustomerID": np.arange(1, rows + 1),
            config.INCOME_COLUMN: [50.0] * rows,
            config.SPENDING_SCORE_COLUMN: [50.0] * rows,
        }
    )
    dataset = clean_dataset(frame)
    with pytest.raises(DataValidationError, match="distinct"):
        validate_clustering_input(dataset)


def test_a_valid_dataset_passes_the_precondition_check(synthetic_frame):
    validate_clustering_input(clean_dataset(synthetic_frame))


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_is_stable_and_content_sensitive(tmp_path, synthetic_frame):
    path = tmp_path / "a.csv"
    synthetic_frame.to_csv(path, index=False)
    first = dataset_fingerprint(path)
    assert first == dataset_fingerprint(path)
    assert len(first) == 64

    changed = synthetic_frame.copy()
    changed.loc[0, config.INCOME_COLUMN] = 999
    other = tmp_path / "b.csv"
    changed.to_csv(other, index=False)
    assert dataset_fingerprint(other) != first
