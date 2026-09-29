"""Loading trusted artifacts and assigning a customer profile to a cluster.

This module is the single inference path. The app, the tests and the command
line all go through :func:`load_artifacts` and :func:`assign_profile`, so there
is no second, slightly different copy of the logic.

Security note: the only file that is ever deserialised is the project-generated
``models/kmeans_pipeline.joblib``, read from a fixed path. No user upload is
ever unpickled, and no request value reaches ``eval``/``exec``/a shell.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .clustering import KMEANS_STEP_NAME, SCALER_STEP_NAME


class ArtifactError(RuntimeError):
    """Raised when the saved model cannot be used."""


class InputValidationError(ValueError):
    """Raised when a customer profile is not a usable pair of numbers."""


@dataclass(frozen=True)
class LoadedArtifacts:
    """A fitted pipeline plus the metadata recorded when it was trained."""

    pipeline: Pipeline
    metadata: dict
    model_path: Path
    metadata_path: Path

    @property
    def n_clusters(self) -> int:
        return int(self.metadata["model"]["selected_k"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["features"]["columns"])

    @property
    def cluster_summary(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.metadata["clusters"])

    def training_range(self) -> dict[str, tuple[float, float]]:
        """Observed min/max of each predictor in the training data."""
        return {
            column: (values["min"], values["max"])
            for column, values in self.metadata["features"]["training_ranges"].items()
        }


@dataclass(frozen=True)
class SegmentAssignment:
    """The outcome of matching one profile to its nearest learned cluster."""

    annual_income_k: float
    spending_score: int
    cluster_id: int
    description: str
    centroid: dict[str, float]
    distances_standardized: dict[int, float]
    nearest_distance_standardized: float
    within_training_range: bool
    out_of_range_notes: list[str]


def get_artifact_paths() -> tuple[Path, Path]:
    """Fixed locations of the trusted artifacts."""
    return (
        config.MODELS_DIR / config.MODEL_FILENAME,
        config.MODELS_DIR / config.METADATA_FILENAME,
    )


def _check_artifact_compatibility(metadata: dict, model_path: Path) -> None:
    version = metadata.get("artifact_schema_version")
    if version != config.ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"{model_path.name} was written with artifact schema version {version}, "
            f"but this code expects {config.ARTIFACT_SCHEMA_VERSION}. "
            "Regenerate the artifacts with: python -m src.train"
        )
    columns = list(metadata.get("features", {}).get("columns", []))
    if columns != list(config.FEATURE_COLUMNS):
        raise ArtifactError(
            f"{model_path.name} was trained on features {columns}, but this code "
            f"expects {list(config.FEATURE_COLUMNS)}. Regenerate the artifacts with: "
            "python -m src.train"
        )


def load_artifacts(
    model_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> LoadedArtifacts:
    """Load the saved pipeline and its metadata.

    Raises:
        ArtifactError: with local setup instructions when something is missing,
            unreadable or was produced by a different version of this project.
    """
    default_model, default_metadata = get_artifact_paths()
    model_file = Path(model_path) if model_path is not None else default_model
    metadata_file = Path(metadata_path) if metadata_path is not None else default_metadata

    missing = [str(p) for p in (model_file, metadata_file) if not p.is_file()]
    if missing:
        raise ArtifactError(
            "Model artifacts are missing or incomplete:\n  - "
            + "\n  - ".join(missing)
            + "\n\nCreate them from a local copy of the dataset with:\n"
            f"  1. Put Mall_Customers.csv in {config.DATA_FILE}\n"
            "  2. Run: python -m src.train\n\n"
            + config.DATASET_SETUP_INSTRUCTIONS
        )

    try:
        pipeline = joblib.load(model_file)
    except Exception as exc:  # noqa: BLE001 - report a friendly message, not a traceback
        raise ArtifactError(
            f"{model_file} could not be loaded ({type(exc).__name__}: {exc}). "
            "The file is most likely from an incompatible scikit-learn version. "
            "Delete models/ and re-run: python -m src.train"
        ) from exc

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"{metadata_file} could not be read as JSON ({exc}). "
            "Re-run: python -m src.train"
        ) from exc

    if not isinstance(pipeline, Pipeline) or KMEANS_STEP_NAME not in pipeline.named_steps:
        raise ArtifactError(
            f"{model_file} does not contain a scaler + KMeans pipeline. "
            "Re-run: python -m src.train"
        )
    _check_artifact_compatibility(metadata, model_file)

    scaler: StandardScaler = pipeline.named_steps[SCALER_STEP_NAME]
    model: KMeans = pipeline.named_steps[KMEANS_STEP_NAME]
    if getattr(scaler, "n_features_in_", None) != len(config.FEATURE_COLUMNS):
        raise ArtifactError(
            f"{model_file} was fitted on "
            f"{getattr(scaler, 'n_features_in_', 'an unknown number of')} feature(s); "
            f"{len(config.FEATURE_COLUMNS)} expected. Re-run: python -m src.train"
        )
    if int(model.n_clusters) != int(metadata["model"]["selected_k"]):
        raise ArtifactError(
            f"{model_file} holds {model.n_clusters} clusters but "
            f"{metadata_file} records selected_k={metadata['model']['selected_k']}. "
            "Re-run: python -m src.train"
        )

    return LoadedArtifacts(
        pipeline=pipeline,
        metadata=metadata,
        model_path=model_file,
        metadata_path=metadata_file,
    )


def _as_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InputValidationError(f"{label} must be a number, got {type(value).__name__}.")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise InputValidationError(f"{label} must be a finite number, got {value!r}.")
    return number


def validate_profile_inputs(annual_income_k: object, spending_score: object) -> tuple[float, int]:
    """Validate one customer profile. This is the only place these rules live.

    Returns the income as a float and the spending score as an int.
    Raises :class:`InputValidationError` with a message meant for the user.
    """
    income = _as_finite_float(annual_income_k, "Annual income (k$)")
    score = _as_finite_float(spending_score, "Spending score (1-100)")

    if income < config.INCOME_MIN:
        raise InputValidationError(
            f"Annual income (k$) must be {config.INCOME_MIN:g} or more, got {income:g}."
        )
    if not score.is_integer():
        raise InputValidationError(
            f"Spending score (1-100) must be a whole number, got {score:g}."
        )
    score_int = int(score)
    if not config.SPENDING_SCORE_MIN <= score_int <= config.SPENDING_SCORE_MAX:
        raise InputValidationError(
            f"Spending score must be between {config.SPENDING_SCORE_MIN:g} and "
            f"{config.SPENDING_SCORE_MAX:g}, got {score_int}."
        )
    return income, score_int


def _training_range_notes(
    income: float, score: int, ranges: dict[str, tuple[float, float]]
) -> list[str]:
    notes: list[str] = []
    for column, value in (
        (config.INCOME_COLUMN, income),
        (config.SPENDING_SCORE_COLUMN, float(score)),
    ):
        low, high = ranges.get(column, (None, None))
        if low is None:
            continue
        if value < low or value > high:
            notes.append(
                f"{config.FEATURE_LABELS[column]} = {value:g} is outside the training "
                f"range {low:g} to {high:g} k$. The nearest cluster is still the closest "
                "one the model knows, but the dataset contains no similar customers, so "
                "treat the match with care."
            )
    return notes


def assign_profile(
    artifacts: LoadedArtifacts,
    annual_income_k: object,
    spending_score: object,
) -> SegmentAssignment:
    """Assign one customer profile to its nearest learned cluster.

    The profile's existing spending score is matched against the clusters that
    were learned from real customers. This does **not** predict future spending.
    """
    income, score = validate_profile_inputs(annual_income_k, spending_score)

    scaler: StandardScaler = artifacts.pipeline.named_steps[SCALER_STEP_NAME]
    model: KMeans = artifacts.pipeline.named_steps[KMEANS_STEP_NAME]

    frame = pd.DataFrame(
        [[income, float(score)]], columns=list(artifacts.feature_columns)
    )
    cluster_id = int(model.predict(scaler.transform(frame))[0])

    standardized = scaler.transform(frame)[0]
    distances = {
        int(i): float(np.linalg.norm(standardized - centre))
        for i, centre in enumerate(model.cluster_centers_)
    }

    summary = artifacts.cluster_summary
    row = summary.loc[summary["cluster_id"] == cluster_id].iloc[0]

    notes = _training_range_notes(income, score, artifacts.training_range())
    return SegmentAssignment(
        annual_income_k=income,
        spending_score=score,
        cluster_id=cluster_id,
        description=str(row[config.METADATA_DESCRIPTION_KEY]),
        centroid={
            config.INCOME_COLUMN: float(row[config.METADATA_CENTROID_INCOME_KEY]),
            config.SPENDING_SCORE_COLUMN: float(row[config.METADATA_CENTROID_SCORE_KEY]),
        },
        distances_standardized=distances,
        nearest_distance_standardized=distances[cluster_id],
        within_training_range=not notes,
        out_of_range_notes=notes,
    )


def assign_profiles(artifacts: LoadedArtifacts, features: pd.DataFrame) -> np.ndarray:
    """Batch version of :func:`assign_profile`, used to check the saved model."""
    ordered = features.loc[:, list(artifacts.feature_columns)]
    return np.asarray(artifacts.pipeline.predict(ordered), dtype=int)


def format_assignment(result: SegmentAssignment, artifacts: LoadedArtifacts) -> str:
    """Plain-text summary of one assignment, used by the command line."""
    lines = [
        "",
        f"Annual income  : {result.annual_income_k:,.1f} k$",
        f"Spending score : {result.spending_score}",
        "",
        f"Cluster        : C{result.cluster_id}",
        f"Description    : {result.description} (relative to this dataset)",
        f"Centroid       : {result.centroid[config.INCOME_COLUMN]:,.1f} k$ income, "
        f"spending score {result.centroid[config.SPENDING_SCORE_COLUMN]:,.1f}",
        f"Distance       : {result.nearest_distance_standardized:.4f} "
        "(standardised feature space - a geometric distance, not a confidence score)",
        "",
        "Distances to every learned centroid (standardised units):",
    ]
    for cluster_id, distance in sorted(result.distances_standardized.items()):
        marker = "  <- nearest" if cluster_id == result.cluster_id else ""
        lines.append(f"  C{cluster_id}: {distance:.4f}{marker}")
    lines += [
        "",
        f"Model: k = {artifacts.n_clusters} clusters, trained "
        f"{artifacts.metadata['trained_at_utc']} from "
        f"{artifacts.metadata['dataset']['local_file']} "
        f"(SHA-256 {artifacts.metadata['dataset']['sha256'][:16]}...).",
        "This assigns an existing spending score to a group. It does not predict",
        "future spending.",
    ]
    for note in result.out_of_range_notes:
        lines.append(f"WARNING: {note}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: ``python -m src.predict --income 82 --score 91``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description=(
            "Assign one customer profile to the nearest learned cluster. "
            "This uses the saved artifacts; it never retrains."
        ),
    )
    parser.add_argument(
        "--income", type=float, required=True, metavar="K$",
        help="Annual income in k$ (thousands), as labelled by the dataset",
    )
    parser.add_argument(
        "--score", type=int, required=True, metavar="1-100",
        help="Spending score, a whole number from 1 to 100",
    )
    args = parser.parse_args(argv)

    try:
        artifacts = load_artifacts()
    except ArtifactError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    try:
        result = assign_profile(artifacts, args.income, args.score)
    except InputValidationError as exc:
        print(f"\nThat profile cannot be used: {exc}\n", file=sys.stderr)
        return 2

    print(format_assignment(result, artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
