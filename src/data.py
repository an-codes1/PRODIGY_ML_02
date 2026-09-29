"""Loading, validating and cleaning the customer profile CSV.

The policy in this module is deliberately conservative and is documented in the
README:

* **Complete cases only.** A row is kept only when both predictors are present,
  numeric and finite, income is non-negative, and the spending score is a whole
  number between 1 and 100. Nothing is imputed, because an invented income or
  spending score would be an invented customer.
* **No deduplication of people.** Two customers with the same income *and* the
  same spending score are two different customers, so they are both kept. The
  number of such coincidences is reported. Only records that are identical in
  *every* column are treated as a duplicated record and dropped, keeping the
  first occurrence.
* **Traceability.** Every kept row keeps its 1-based position in the source CSV
  plus the dataset's own ``CustomerID``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when the dataset cannot be used for the clustering task."""


@dataclass(frozen=True)
class CleaningReport:
    """Plain-data summary of what happened to the raw file."""

    rows_total: int
    rows_kept: int
    rows_excluded: int
    exact_duplicate_rows_removed: int
    duplicate_feature_pairs: int
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    columns_present: list[str] = field(default_factory=list)
    missing_required_columns: list[str] = field(default_factory=list)
    feature_ranges: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rows_total": self.rows_total,
            "rows_kept": self.rows_kept,
            "rows_excluded": self.rows_excluded,
            "exact_duplicate_rows_removed": self.exact_duplicate_rows_removed,
            "duplicate_feature_pairs": self.duplicate_feature_pairs,
            "exclusion_reasons": dict(self.exclusion_reasons),
            "columns_present": list(self.columns_present),
            "missing_required_columns": list(self.missing_required_columns),
            "feature_ranges": {k: dict(v) for k, v in self.feature_ranges.items()},
        }


@dataclass(frozen=True)
class CleanedDataset:
    """The rows that survived cleaning, in source order, plus the report."""

    frame: pd.DataFrame
    report: CleaningReport

    @property
    def features(self) -> pd.DataFrame:
        """The two clustering columns, in the fixed order the model expects."""
        return self.frame.loc[:, list(config.FEATURE_COLUMNS)]

    @property
    def row_count(self) -> int:
        return int(len(self.frame))

    @property
    def distinct_feature_pairs(self) -> int:
        return int(self.features.drop_duplicates().shape[0])


def dataset_fingerprint(csv_path: Path | None = None) -> str:
    """SHA-256 of the CSV file, so a result can be tied to an exact input file."""
    path = Path(csv_path) if csv_path is not None else config.DATA_FILE
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def check_required_columns(frame: pd.DataFrame) -> list[str]:
    """Return the required columns that are missing (empty list means fine)."""
    return [name for name in config.REQUIRED_COLUMNS if name not in frame.columns]


def load_dataset(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV and check that the required columns exist.

    Raises:
        FileNotFoundError: with manual download instructions when the file is absent.
        DataValidationError: when a required column is missing.
    """
    path = Path(csv_path) if csv_path is not None else config.DATA_FILE
    if not path.is_file():
        raise FileNotFoundError(config.DATASET_SETUP_INSTRUCTIONS)

    frame = pd.read_csv(path)

    missing = check_required_columns(frame)
    if missing:
        raise DataValidationError(
            "The dataset is missing required column(s): "
            + ", ".join(missing)
            + f". Found columns: {list(frame.columns)}. "
            "Expected at least "
            + ", ".join(config.REQUIRED_COLUMNS)
            + "."
        )
    return frame


def _row_problems(income: pd.Series, score: pd.Series) -> pd.DataFrame:
    """One boolean column per rule. A row is excluded if any rule is True."""
    income_finite = income.notna() & np.isfinite(income)
    score_finite = score.notna() & np.isfinite(score)
    return pd.DataFrame(
        {
            "income_not_numeric_or_missing": ~income_finite,
            "spending_score_not_numeric_or_missing": ~score_finite,
            "income_negative": income_finite & (income < config.INCOME_MIN),
            "spending_score_out_of_range": score_finite
            & ((score < config.SPENDING_SCORE_MIN) | (score > config.SPENDING_SCORE_MAX)),
            # The source column is a whole-number 1-100 scale, and the app and the
            # command line both expect a whole number, so it is checked here too.
            "spending_score_not_whole_number": score_finite & (score % 1 != 0),
        },
        index=income.index,
    )


def clean_dataset(
    frame: pd.DataFrame,
    *,
    drop_exact_duplicates: bool = True,
) -> CleanedDataset:
    """Apply the documented complete-case policy and build a cleaning report.

    Args:
        frame: raw rows straight from :func:`load_dataset`.
        drop_exact_duplicates: remove records identical in every column
            (keeping the first). Duplicate *feature pairs* are never removed.
    """
    missing = check_required_columns(frame)
    if missing:
        raise DataValidationError(
            "Cannot clean the dataset, required column(s) missing: " + ", ".join(missing)
        )

    rows_total = int(len(frame))
    work = frame.copy()

    # 1-based row number in the CSV data section (header excluded) for tracing.
    work[config.SOURCE_ROW_COLUMN] = np.arange(1, rows_total + 1)

    # Try hard to read the predictors as numbers. Anything that will not convert
    # becomes NaN and is then reported as an exclusion reason.
    for column in config.FEATURE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    problems = _row_problems(work[config.INCOME_COLUMN], work[config.SPENDING_SCORE_COLUMN])
    exclusion_reasons = {
        rule: int(count) for rule, count in problems.sum().items() if count > 0
    }

    valid = ~problems.any(axis=1)
    kept = work.loc[valid].copy()

    # Only records identical in *every* column count as a duplicated record.
    duplicate_rows_removed = 0
    if drop_exact_duplicates and not kept.empty:
        all_columns = [column for column in frame.columns]
        duplicate_mask = kept.duplicated(subset=all_columns, keep="first")
        duplicate_rows_removed = int(duplicate_mask.sum())
        kept = kept.loc[~duplicate_mask].copy()

    # Report, but never drop, customers who share an income + spending score.
    duplicate_feature_pairs = (
        int(kept.duplicated(subset=list(config.FEATURE_COLUMNS)).sum())
        if not kept.empty
        else 0
    )

    kept = kept.reset_index(drop=True)

    feature_ranges: dict[str, dict[str, float]] = {}
    if not kept.empty:
        for column in config.FEATURE_COLUMNS:
            values = kept[column]
            feature_ranges[column] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }

    report = CleaningReport(
        rows_total=rows_total,
        rows_kept=int(len(kept)),
        rows_excluded=rows_total - int(len(kept)),
        exact_duplicate_rows_removed=duplicate_rows_removed,
        duplicate_feature_pairs=duplicate_feature_pairs,
        exclusion_reasons=exclusion_reasons,
        columns_present=[str(c) for c in frame.columns],
        missing_required_columns=missing,
        feature_ranges=feature_ranges,
    )
    return CleanedDataset(frame=kept, report=report)


def validate_clustering_input(dataset: CleanedDataset) -> None:
    """Refuse to cluster data that cannot produce meaningful metrics.

    Raises:
        DataValidationError: with the reason spelled out.
    """
    if dataset.row_count < config.MIN_ROWS_FOR_CLUSTERING:
        raise DataValidationError(
            f"Only {dataset.row_count} valid row(s) were available; at least "
            f"{config.MIN_ROWS_FOR_CLUSTERING} are needed before clustering results "
            "would be meaningful."
        )
    if dataset.distinct_feature_pairs < config.MIN_DISTINCT_FEATURE_PAIRS:
        raise DataValidationError(
            f"Only {dataset.distinct_feature_pairs} distinct "
            f"{'/'.join(config.FEATURE_COLUMNS)} pair(s) were found; at least "
            f"{config.MIN_DISTINCT_FEATURE_PAIRS} are needed. The predictors are "
            "probably constant or nearly constant."
        )
    features = dataset.features
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise DataValidationError(
            "Predictor values contain NaN or infinity after cleaning. This is a bug "
            "in the cleaning step, not a data problem."
        )
