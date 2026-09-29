"""K-means evaluation, k selection and cluster summaries.

Why scale first: K-means minimises the sum of squared **distances**, and squared
distance weights a column by its variance. :class:`sklearn.preprocessing.
StandardScaler` gives both features comparable variance - it subtracts each
feature's mean and divides by its standard deviation - so their numerical scales
do not disproportionately influence distances. Whether an unscaled feature would
dominate depends on its actual numerical spread, not on what its unit is called.
In this dataset the two spreads happen to be similar, so scaling is a safeguard
here rather than a correction; on data whose columns are recorded on very
different scales it changes the answer substantially.

Everything is fitted once on the standardised matrix and the silhouette score
is measured in that same matrix, so "inertia" and "silhouette" describe the
clustering that the model actually learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data import CleanedDataset, DataValidationError

KMEANS_STEP_NAME = "kmeans"
SCALER_STEP_NAME = "scaler"


def build_pipeline(
    k: int,
    *,
    random_state: int = config.RANDOM_STATE,
    n_init: int = config.N_INIT,
) -> Pipeline:
    """Create the shared scaler + K-means pipeline.

    The same object is used for training, for the elbow/silhouette sweep, for
    the saved artifacts and for the app, so there is exactly one code path.
    """
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise DataValidationError(f"k must be an integer, got {type(k).__name__}.")
    if k < 1:
        raise DataValidationError(f"k must be 1 or more, got {k}.")
    if n_init < 1:
        raise DataValidationError(f"n_init must be 1 or more, got {n_init}.")

    return Pipeline(
        steps=[
            (SCALER_STEP_NAME, StandardScaler()),
            (
                KMEANS_STEP_NAME,
                KMeans(
                    n_clusters=int(k),
                    random_state=random_state,
                    n_init=int(n_init),
                ),
            ),
        ]
    )


def _max_valid_k(n_rows: int, n_distinct: int, requested_max: int) -> int:
    """Largest k that can possibly be evaluated for this dataset."""
    # A silhouette score needs at least 2 clusters and fewer clusters than rows.
    return int(max(0, min(requested_max, n_rows - 1, n_distinct)))


def evaluate_candidate_ks(
    features: pd.DataFrame,
    *,
    k_min: int = config.K_MIN,
    k_max: int = config.K_MAX,
    random_state: int = config.RANDOM_STATE,
    n_init: int = config.N_INIT,
) -> pd.DataFrame:
    """Fit K-means for k = ``k_min``..``k_max`` and score each candidate.

    Returns a table with one row per candidate k and the columns
    ``k``, ``n_clusters_found``, ``inertia``, ``silhouette``, ``silhouette_valid``
    and ``note``. A candidate that cannot be scored (k out of range, or a fit
    that collapsed into fewer groups than requested) keeps its inertia but has
    ``silhouette = None`` so the selector can skip it.
    """
    if k_min < 1:
        raise DataValidationError(f"k_min must be 1 or more, got {k_min}.")
    if k_max < k_min:
        raise DataValidationError(f"k_max ({k_max}) must not be below k_min ({k_min}).")

    matrix = features.loc[:, list(config.FEATURE_COLUMNS)].to_numpy(dtype=float)
    n_rows, n_distinct = matrix.shape[0], int(np.unique(matrix, axis=0).shape[0])

    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)

    hard_max = _max_valid_k(n_rows, n_distinct, k_max)

    records: list[dict] = []
    for k in range(k_min, k_max + 1):
        record: dict = {
            "k": int(k),
            "n_clusters_found": 0,
            "inertia": None,
            "silhouette": None,
            "silhouette_valid": False,
            "note": "",
        }

        if k > n_rows:
            record["note"] = f"skipped: k={k} exceeds the {n_rows} available rows"
            records.append(record)
            continue

        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(scaled)
        found = int(len(np.unique(labels)))
        record["n_clusters_found"] = found
        record["inertia"] = float(model.inertia_)

        if k < config.SILHOUETTE_K_MIN:
            record["note"] = "silhouette needs at least 2 clusters"
        elif k > hard_max:
            record["note"] = (
                f"skipped: k={k} exceeds the {hard_max} clusters this dataset can "
                "separate (row count and distinct feature pairs)"
            )
        elif found < 2:
            record["note"] = "skipped: the fit collapsed into fewer than 2 groups"
        else:
            try:
                record["silhouette"] = float(silhouette_score(scaled, labels))
                record["silhouette_valid"] = True
            except ValueError as exc:  # pragma: no cover - defensive
                record["note"] = f"skipped: silhouette could not be computed ({exc})"
        records.append(record)

    return pd.DataFrame.from_records(records)


def select_k(candidates: pd.DataFrame) -> int:
    """Pick the k with the highest measured silhouette score.

    Exact ties go to the smaller k, because fewer groups is the simpler story
    when the evidence is the same.
    """
    if "silhouette" not in candidates.columns:
        raise DataValidationError("Candidate table has no 'silhouette' column.")

    scored = candidates.loc[candidates["silhouette"].notna()]
    if scored.empty:
        raise DataValidationError(
            "No candidate k produced a valid silhouette score, so k cannot be "
            "selected. Check the candidate table and the dataset size."
        )

    best = float(scored["silhouette"].max())
    tied = scored.loc[np.isclose(scored["silhouette"], best, rtol=0.0, atol=1e-12), "k"]
    return int(min(tied))


def fit_pipeline(
    features: pd.DataFrame,
    k: int,
    *,
    random_state: int = config.RANDOM_STATE,
    n_init: int = config.N_INIT,
) -> tuple[Pipeline, np.ndarray]:
    """Fit the scaler + K-means pipeline and return it with the training labels."""
    pipeline = build_pipeline(k, random_state=random_state, n_init=n_init)
    ordered = features.loc[:, list(config.FEATURE_COLUMNS)]
    pipeline.fit(ordered)
    labels = pipeline.named_steps[KMEANS_STEP_NAME].predict(
        pipeline.named_steps[SCALER_STEP_NAME].transform(ordered)
    )
    return pipeline, np.asarray(labels, dtype=int)


def centroids_in_original_units(pipeline: Pipeline) -> pd.DataFrame:
    """Convert the learned (standardised) centroids back to original units."""
    scaler: StandardScaler = pipeline.named_steps[SCALER_STEP_NAME]
    model: KMeans = pipeline.named_steps[KMEANS_STEP_NAME]
    original = scaler.inverse_transform(model.cluster_centers_)
    frame = pd.DataFrame(original, columns=list(config.FEATURE_COLUMNS))
    frame.insert(0, "cluster_id", list(range(len(frame))))
    return frame


def describe_cluster(
    income: float,
    score: float,
    reference_income: float,
    reference_score: float,
    *,
    tolerance: float = config.RELATIVE_TOLERANCE,
) -> str:
    """Build a neutral description of one centroid.

    The wording is always relative to this dataset - for example "higher income,
    lower spending score". It says nothing about age, gender, loyalty or why a
    customer spends the way they do, because the model never saw those columns.
    """

    def band(value: float, reference: float, unit: str) -> str:
        if reference == 0:
            return f"mid-range {unit}"
        ratio = value / reference
        if ratio > 1.0 + tolerance:
            return f"higher {unit}"
        if ratio < 1.0 - tolerance:
            return f"lower {unit}"
        return f"mid-range {unit}"

    return (
        f"{band(income, reference_income, 'income')}, "
        f"{band(score, reference_score, 'spending score')}"
    )


def build_cluster_summary(
    features: pd.DataFrame,
    labels: np.ndarray,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """One row per cluster: count, centroid, observed means, description, colour.

    The column names are the ones in :mod:`src.config`, so this DataFrame, the
    saved metadata, the CSV exports and the app all speak the same language. The
    order follows the cluster id, so every output in the project lines up.
    """
    reference_income = float(features[config.INCOME_COLUMN].mean())
    reference_score = float(features[config.SPENDING_SCORE_COLUMN].mean())

    records = []
    for cluster_id in sorted({int(v) for v in labels}):
        member = features.loc[labels == cluster_id]
        centroid = centroids.loc[centroids["cluster_id"] == cluster_id].iloc[0]
        centroid_income = float(centroid[config.INCOME_COLUMN])
        centroid_score = float(centroid[config.SPENDING_SCORE_COLUMN])
        records.append(
            {
                config.METADATA_CLUSTER_ID_KEY: cluster_id,
                config.METADATA_CUSTOMERS_KEY: int(len(member)),
                "share_of_customers": float(len(member) / len(features)),
                config.METADATA_CENTROID_INCOME_KEY: centroid_income,
                config.METADATA_CENTROID_SCORE_KEY: centroid_score,
                config.METADATA_MEAN_INCOME_KEY: float(
                    member[config.INCOME_COLUMN].mean()
                ),
                config.METADATA_MEAN_SCORE_KEY: float(
                    member[config.SPENDING_SCORE_COLUMN].mean()
                ),
                config.METADATA_DESCRIPTION_KEY: describe_cluster(
                    centroid_income, centroid_score, reference_income, reference_score
                ),
                config.METADATA_COLOR_KEY: config.CLUSTER_COLORS[
                    cluster_id % len(config.CLUSTER_COLORS)
                ],
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_dataset(dataset: CleanedDataset) -> dict[str, float]:
    """Mean of each predictor, used as the reference for the descriptions."""
    return {
        column: float(dataset.features[column].mean())
        for column in config.FEATURE_COLUMNS
    }
