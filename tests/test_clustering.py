"""K-means evaluation, k selection, scaling and cluster descriptions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.clustering import (
    KMEANS_STEP_NAME,
    SCALER_STEP_NAME,
    build_cluster_summary,
    build_pipeline,
    centroids_in_original_units,
    describe_cluster,
    evaluate_candidate_ks,
    fit_pipeline,
    select_k,
)
from src.data import clean_dataset


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def test_pipeline_is_scaler_then_kmeans_with_the_fixed_settings():
    pipeline = build_pipeline(4)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == [SCALER_STEP_NAME, KMEANS_STEP_NAME]
    assert isinstance(pipeline.named_steps[SCALER_STEP_NAME], StandardScaler)

    model = pipeline.named_steps[KMEANS_STEP_NAME]
    assert isinstance(model, KMeans)
    assert model.n_clusters == 4
    assert model.random_state == config.RANDOM_STATE == 42
    assert model.n_init == config.N_INIT == 20


@pytest.mark.parametrize("bad_k", [0, -3, 2.5, "3", True, None])
def test_invalid_k_is_rejected(bad_k):
    with pytest.raises(Exception):
        build_pipeline(bad_k)


def test_invalid_n_init_is_rejected():
    with pytest.raises(Exception):
        build_pipeline(3, n_init=0)


# ---------------------------------------------------------------------------
# Feature order and scaling
# ---------------------------------------------------------------------------
def test_features_are_read_in_the_declared_order(synthetic_frame):
    shuffled = synthetic_frame.loc[
        :,
        [config.SPENDING_SCORE_COLUMN, "Age", config.INCOME_COLUMN, "CustomerID"],
    ]
    ordered = synthetic_frame.loc[:, list(config.FEATURE_COLUMNS)]

    a = fit_pipeline(shuffled, 3)[1]
    b = fit_pipeline(ordered, 3)[1]
    assert np.array_equal(a, b)


def test_fitted_scaler_is_standardised(messy_frame):
    dataset = clean_dataset(messy_frame)
    pipeline, _ = fit_pipeline(dataset.features, 3)

    scaler = pipeline.named_steps[SCALER_STEP_NAME]
    transformed = scaler.transform(dataset.features)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-10)


def test_clustering_ignores_the_identifier_and_demographic_columns(synthetic_frame):
    relabelled = synthetic_frame.copy()
    relabelled["CustomerID"] = relabelled["CustomerID"] * 1000 + 17
    relabelled["Age"] = 70 - relabelled["Age"]
    relabelled["Gender"] = "Female"

    original_labels = fit_pipeline(synthetic_frame, 4)[1]
    changed_labels = fit_pipeline(relabelled, 4)[1]
    assert np.array_equal(original_labels, changed_labels)


def test_scaling_matters_because_units_differ():
    """Distance-based clustering is only fair once the units are comparable.

    Here the two columns carry genuinely different *numerical spread*: income
    varies by thousands while spending score varies by tens. Squared distance
    weights a column by its variance, so on the raw columns K-means chases the
    income noise and misses the two spending-score groups. Standardising gives
    both columns comparable variance and recovers the real structure.

    Note what this test is and is not: the effect comes from the variance
    difference, not from the unit *names*. Nothing about "dollars" versus "k$"
    matters to K-means - only the numbers do. See
    ``test_scaling_is_harmless_when_spreads_are_similar`` for the other side.

    Cluster ids are arbitrary, so the partitions are compared with the adjusted
    Rand index rather than by matching numbers.
    """
    rng = np.random.default_rng(3)
    per_group = 60
    rows = 2 * per_group
    true_group = np.repeat([0, 1], per_group)
    frame = pd.DataFrame(
        {
            config.INCOME_COLUMN: 60_000 + rng.normal(0, 2_000, rows),
            config.SPENDING_SCORE_COLUMN: np.repeat([10.0, 90.0], per_group)
            + rng.normal(0, 1, rows),
        }
    )

    # Guard the premise: the two columns really do differ in spread.
    spreads = frame.std(ddof=0)
    assert spreads[config.INCOME_COLUMN] / spreads[config.SPENDING_SCORE_COLUMN] > 50

    scaled_labels = fit_pipeline(frame, 2)[1]
    raw_labels = KMeans(
        n_clusters=2, random_state=config.RANDOM_STATE, n_init=config.N_INIT
    ).fit_predict(frame)

    assert adjusted_rand_score(scaled_labels, true_group) > 0.95
    assert adjusted_rand_score(raw_labels, true_group) < 0.2


def test_scaling_is_harmless_when_spreads_are_similar():
    """The flip side: comparable spread means scaling changes nothing.

    This is the honest counterweight to the test above. Whether an unscaled
    feature dominates depends on its actual numerical spread, not on what its
    unit is called. When two columns already spread over comparable ranges,
    the raw and standardised fits agree, and the app's own columns are in
    exactly this situation (standard deviations of roughly 26 and 26).
    """
    rng = np.random.default_rng(11)
    per_group = 50
    rows = 2 * per_group
    true_group = np.repeat([0, 1], per_group)

    # Same structure, but both columns scaled to a similar spread.
    frame = pd.DataFrame(
        {
            config.INCOME_COLUMN: 55 + rng.normal(0, 20, rows),
            config.SPENDING_SCORE_COLUMN: np.repeat([20.0, 80.0], per_group)
            + rng.normal(0, 6, rows),
        }
    )

    spreads = frame.std(ddof=0)
    ratio = spreads[config.INCOME_COLUMN] / spreads[config.SPENDING_SCORE_COLUMN]
    assert 0.5 < ratio < 2.0, f"premise broken, spread ratio was {ratio:.2f}"

    raw_labels = KMeans(
        n_clusters=2, random_state=config.RANDOM_STATE, n_init=config.N_INIT
    ).fit_predict(frame)
    scaled_labels = fit_pipeline(frame, 2)[1]

    # Both find the real groups, and they agree with each other.
    assert adjusted_rand_score(raw_labels, true_group) > 0.9
    assert adjusted_rand_score(scaled_labels, raw_labels) > 0.9


def test_standard_scaler_subtracts_the_mean_and_divides_by_the_std(messy_frame):
    """The scaler must be the plain mean/standard-deviation transform.

    Asserted numerically rather than by checking any sentence about it.
    """
    dataset = clean_dataset(messy_frame)
    pipeline, _ = fit_pipeline(dataset.features, 3)
    scaler = pipeline.named_steps["scaler"]

    expected = (
        (dataset.features - dataset.features.mean()) / dataset.features.std(ddof=0)
    )
    got = scaler.transform(dataset.features)

    assert np.allclose(scaler.mean_, dataset.features.mean().to_numpy())
    assert np.allclose(scaler.scale_, dataset.features.std(ddof=0).to_numpy())
    assert np.allclose(got, expected.to_numpy())


# ---------------------------------------------------------------------------
# Candidate sweep
# ---------------------------------------------------------------------------
def test_candidates_cover_the_requested_range_and_skip_invalid_k(synthetic_frame):
    dataset = clean_dataset(synthetic_frame)
    candidates = evaluate_candidate_ks(dataset.features, k_max=10)

    assert candidates["k"].tolist() == list(range(1, 11))
    assert candidates["inertia"].notna().all()
    # k = 1 cannot have a silhouette score.
    assert pd.isna(candidates.loc[candidates["k"] == 1, "silhouette"].iloc[0])
    assert candidates.loc[candidates["k"] >= 2, "silhouette"].notna().all()
    # Inertia must fall as k rises.
    assert candidates["inertia"].is_monotonic_decreasing


def test_silhouette_is_measured_in_the_standardised_space(synthetic_frame):
    dataset = clean_dataset(synthetic_frame)
    candidates = evaluate_candidate_ks(dataset.features, k_max=6)

    scaler = StandardScaler().fit(dataset.features)
    scaled = scaler.transform(dataset.features)
    model = KMeans(
        n_clusters=3, random_state=config.RANDOM_STATE, n_init=config.N_INIT
    ).fit(scaled)
    from sklearn.metrics import silhouette_score

    expected = silhouette_score(scaled, model.labels_)
    recorded = candidates.loc[candidates["k"] == 3, "silhouette"].iloc[0]
    assert recorded == pytest.approx(expected)


def test_k_above_the_sample_size_is_skipped_not_crashed(synthetic_frame):
    dataset = clean_dataset(synthetic_frame)
    candidates = evaluate_candidate_ks(dataset.features, k_max=200)
    beyond = candidates[candidates["k"] > dataset.row_count]
    assert len(beyond) > 0
    assert beyond["inertia"].isna().all()
    assert beyond["note"].str.contains("exceeds").all()


def test_k_max_below_k_min_is_rejected(synthetic_frame):
    dataset = clean_dataset(synthetic_frame)
    with pytest.raises(Exception):
        evaluate_candidate_ks(dataset.features, k_min=3, k_max=2)


def test_degenerate_data_cannot_produce_a_scored_candidate():
    """Every customer identical: no split exists, so no k can be scored."""
    rows = 40
    frame = pd.DataFrame(
        {
            config.INCOME_COLUMN: [50.0] * rows,
            config.SPENDING_SCORE_COLUMN: [50.0] * rows,
        }
    )
    candidates = evaluate_candidate_ks(frame, k_max=6)
    assert candidates["silhouette"].isna().all()
    assert candidates.loc[candidates["k"] >= 2, "note"].str.contains("distinct").all()

    with pytest.raises(Exception, match="valid silhouette"):
        select_k(candidates)


def test_fewer_distinct_pairs_than_k_is_skipped_cleanly():
    """Only two distinct feature pairs, so k above 2 is not scored."""
    rows = 30
    frame = pd.DataFrame(
        {
            config.INCOME_COLUMN: [50.0, 80.0] * (rows // 2),
            config.SPENDING_SCORE_COLUMN: [20.0, 90.0] * (rows // 2),
        }
    )
    candidates = evaluate_candidate_ks(frame, k_max=5)
    assert not pd.isna(candidates.loc[candidates["k"] == 2, "silhouette"].iloc[0])
    assert candidates.loc[candidates["k"] > 2, "silhouette"].isna().all()
    assert select_k(candidates) == 2


# ---------------------------------------------------------------------------
# k selection
# ---------------------------------------------------------------------------
def test_highest_silhouette_wins():
    candidates = pd.DataFrame(
        {
            "k": [1, 2, 3, 4, 5],
            "inertia": [400.0, 270.0, 160.0, 110.0, 65.0],
            "silhouette": [np.nan, 0.32, 0.47, 0.49, 0.55],
        }
    )
    assert select_k(candidates) == 5


def test_exact_ties_go_to_the_smaller_k():
    candidates = pd.DataFrame(
        {
            "k": [2, 3, 4],
            "inertia": [200.0, 150.0, 120.0],
            "silhouette": [0.5, 0.5, 0.49],
        }
    )
    assert select_k(candidates) == 2


def test_a_barely_lower_silhouette_still_loses():
    candidates = pd.DataFrame(
        {
            "k": [2, 3],
            "inertia": [200.0, 150.0],
            "silhouette": [0.5500, 0.5499],
        }
    )
    assert select_k(candidates) == 2


def test_select_k_needs_a_silhouette_column():
    with pytest.raises(Exception, match="silhouette"):
        select_k(pd.DataFrame({"k": [2, 3]}))


# ---------------------------------------------------------------------------
# Centroids, summaries and descriptions
# ---------------------------------------------------------------------------
def test_centroids_are_returned_in_original_units(synthetic_frame):
    dataset = clean_dataset(synthetic_frame)
    pipeline, _ = fit_pipeline(dataset.features, 3)
    centroids = centroids_in_original_units(pipeline)

    assert list(centroids.columns) == [
        "cluster_id",
        config.INCOME_COLUMN,
        config.SPENDING_SCORE_COLUMN,
    ]
    assert centroids["cluster_id"].tolist() == [0, 1, 2]
    # Standardised centres are the origin, so inverting must land inside the data.
    for column in config.FEATURE_COLUMNS:
        assert centroids[column].between(dataset.features[column].min() * 0.9,
                                         dataset.features[column].max() * 1.1).all()


def test_cluster_counts_sum_to_the_cleaned_row_count(synthetic_frame):
    dataset = clean_dataset(synthetic_frame)
    pipeline, labels = fit_pipeline(dataset.features, 3)
    summary = build_cluster_summary(
        dataset.features, labels, centroids_in_original_units(pipeline)
    )
    assert int(summary["customers"].sum()) == dataset.row_count
    assert summary["cluster_id"].tolist() == sorted(summary["cluster_id"].tolist())
    assert summary["share_of_customers"].sum() == pytest.approx(1.0)


def test_descriptions_are_relative_and_neutral():
    assert describe_cluster(100.0, 80.0, 60.0, 50.0) == "higher income, higher spending score"
    assert describe_cluster(30.0, 20.0, 60.0, 50.0) == "lower income, lower spending score"
    assert describe_cluster(30.0, 90.0, 60.0, 50.0) == "lower income, higher spending score"
    assert describe_cluster(60.0, 50.0, 60.0, 50.0) == (
        "mid-range income, mid-range spending score"
    )
    # Just inside the 10% band still counts as mid-range.
    assert describe_cluster(65.0, 50.0, 60.0, 50.0).startswith("mid-range income")


def test_summary_colours_follow_the_cluster_id(synthetic_frame):
    dataset = clean_dataset(synthetic_frame)
    pipeline, labels = fit_pipeline(dataset.features, 4)
    summary = build_cluster_summary(
        dataset.features, labels, centroids_in_original_units(pipeline)
    )
    for row in summary.to_dict(orient="records"):
        assert row["color"] == config.CLUSTER_COLORS[row["cluster_id"] % len(config.CLUSTER_COLORS)]
