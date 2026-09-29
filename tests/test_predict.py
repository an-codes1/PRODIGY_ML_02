"""Artifact loading, single-profile validation and assignment consistency."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src import config
from src.clustering import fit_pipeline
from src.data import clean_dataset, load_dataset
from src.predict import (
    ArtifactError,
    InputValidationError,
    assign_profile,
    assign_profiles,
    get_artifact_paths,
    load_artifacts,
    validate_profile_inputs,
)


# ---------------------------------------------------------------------------
# Single-profile input validation (the shared, authoritative check)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "income, score",
    [(60.0, 50), (0, 1), (137, 100), (15.5, 99.0)],
)
def test_valid_profiles_are_accepted(income, score):
    assert validate_profile_inputs(income, score) == (float(income), int(score))


@pytest.mark.parametrize(
    "income, score, expected",
    [
        (-1, 50, "must be 0 or more"),
        (60, 0, "must be between 1 and 100"),
        (60, 101, "must be between 1 and 100"),
        (60, 50.5, "whole number"),
        (None, 50, "must be a number"),
        (60, "50", "must be a number"),
        (float("nan"), 50, "finite"),
        (60, float("inf"), "finite"),
        (True, 50, "must be a number"),
    ],
)
def test_invalid_profiles_are_rejected(income, score, expected):
    with pytest.raises(InputValidationError, match=expected):
        validate_profile_inputs(income, score)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def test_missing_artifacts_give_setup_instructions(tmp_path):
    with pytest.raises(ArtifactError) as excinfo:
        load_artifacts(tmp_path / "nope.joblib", tmp_path / "nope.json")
    message = str(excinfo.value)
    assert "python -m src.train" in message
    assert "Mall_Customers.csv" in message


def test_a_non_pipeline_artifact_is_refused(tmp_path, trained_dirs):
    import joblib

    models_dir, _ = trained_dirs
    bad_model = tmp_path / "bad.joblib"
    joblib.dump({"not": "a pipeline"}, bad_model)

    with pytest.raises(ArtifactError, match="pipeline"):
        load_artifacts(bad_model, models_dir / config.METADATA_FILENAME)


def test_metadata_from_another_schema_version_is_refused(trained_dirs):
    models_dir, _ = trained_dirs
    metadata_file = models_dir / config.METADATA_FILENAME
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["artifact_schema_version"] = 999
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="schema version"):
        load_artifacts(models_dir / config.MODEL_FILENAME, metadata_file)


def test_metadata_with_different_features_is_refused(trained_dirs):
    models_dir, _ = trained_dirs
    metadata_file = models_dir / config.METADATA_FILENAME
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["features"]["columns"] = ["Age", config.INCOME_COLUMN]
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="trained on features"):
        load_artifacts(models_dir / config.MODEL_FILENAME, metadata_file)


def test_a_truncated_model_file_gives_a_friendly_message(tmp_path, trained_dirs):
    models_dir, _ = trained_dirs
    broken = tmp_path / "broken.joblib"
    shutil.copyfile(models_dir / config.MODEL_FILENAME, broken)
    broken.write_bytes(broken.read_bytes()[:120])

    with pytest.raises(ArtifactError, match="could not be loaded"):
        load_artifacts(broken, models_dir / config.METADATA_FILENAME)


def test_default_artifact_paths_are_inside_the_project():
    model_path, metadata_path = get_artifact_paths()
    assert model_path == config.MODELS_DIR / config.MODEL_FILENAME
    assert metadata_path == config.MODELS_DIR / config.METADATA_FILENAME
    assert config.PROJECT_ROOT in model_path.parents


# ---------------------------------------------------------------------------
# Save / load equivalence
# ---------------------------------------------------------------------------
def test_saved_pipeline_predicts_exactly_like_the_fitted_one(
    trained_dirs, trained_artifacts, synthetic_csv
):
    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    dataset = clean_dataset(load_dataset(synthetic_csv))
    assert np.array_equal(
        assign_profiles(artifacts, dataset.features),
        trained_artifacts.segments["cluster_id"].to_numpy(),
    )
    # The saved scaler is the fitted one, not a fresh fit.
    assert np.allclose(
        artifacts.pipeline.named_steps["scaler"].mean_, dataset.features.mean().to_numpy()
    )
    assert np.allclose(
        artifacts.pipeline.named_steps["scaler"].var_, dataset.features.var(ddof=0).to_numpy()
    )


def test_loaded_model_never_refits_on_prediction(trained_dirs, synthetic_csv):
    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    before = artifacts.pipeline.named_steps["kmeans"].cluster_centers_.copy()
    dataset = clean_dataset(load_dataset(synthetic_csv))
    assign_profiles(artifacts, dataset.features)
    assert np.array_equal(before, artifacts.pipeline.named_steps["kmeans"].cluster_centers_)


# ---------------------------------------------------------------------------
# Assignment consistency
# ---------------------------------------------------------------------------
def test_assignment_matches_a_direct_pipeline_predict(trained_dirs):
    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    result = assign_profile(artifacts, 82, 91)
    frame = pd.DataFrame(
        [[82.0, 91.0]], columns=list(config.FEATURE_COLUMNS)
    )
    assert result.cluster_id == int(artifacts.pipeline.predict(frame)[0])


def test_cluster_id_description_and_centroid_agree_with_the_summary(trained_dirs):
    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    summary = artifacts.cluster_summary
    for income, score in [(25, 80), (90, 15), (55, 50), (130, 99), (15, 1)]:
        result = assign_profile(artifacts, income, score)
        row = summary.loc[summary["cluster_id"] == result.cluster_id].iloc[0]
        assert result.description == row["description"]
        assert result.centroid[config.INCOME_COLUMN] == pytest.approx(
            row["centroid_income_k"]
        )
        assert result.centroid[config.SPENDING_SCORE_COLUMN] == pytest.approx(
            row["centroid_spending_score"]
        )
        assert result.nearest_distance_standardized == pytest.approx(
            min(result.distances_standardized.values())
        )


def test_distances_cover_every_learned_cluster(trained_dirs):
    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    result = assign_profile(artifacts, 60, 50)
    assert sorted(result.distances_standardized) == list(range(artifacts.n_clusters))
    assert all(value >= 0 for value in result.distances_standardized.values())


def test_out_of_range_profile_gets_a_warning(trained_dirs):
    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    ranges = artifacts.training_range()
    high_income, high_score = ranges[config.INCOME_COLUMN][1] + 200, 100

    result = assign_profile(artifacts, high_income, high_score)
    assert result.within_training_range is False
    assert result.out_of_range_notes
    assert any("outside the training range" in note for note in result.out_of_range_notes)


def test_in_range_profile_gets_no_warning(trained_dirs):
    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    result = assign_profile(artifacts, 60, 50)
    assert result.within_training_range is True
    assert result.out_of_range_notes == []


def test_assign_profile_validates_before_using_the_model(trained_dirs):
    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    with pytest.raises(InputValidationError):
        assign_profile(artifacts, 60, 250)


# ---------------------------------------------------------------------------
# Real artifacts, when they exist
# ---------------------------------------------------------------------------
def test_real_training_counts_match_the_cleaned_rows(real_artifacts, real_metadata):
    cleaning = real_metadata["dataset"]["cleaning"]
    assert sum(c["customers"] for c in real_metadata["clusters"]) == cleaning["rows_kept"]
    assert cleaning["rows_kept"] + cleaning["rows_excluded"] == cleaning["rows_total"]


def test_real_inference_labels_match_the_fitted_model(real_artifacts, real_dataset):
    from src.predict import assign_profiles

    labels = assign_profiles(real_artifacts, real_dataset.features)
    assert len(labels) == real_dataset.row_count
    assert set(labels.tolist()) == set(range(real_artifacts.n_clusters))


def test_real_metadata_is_consistent_json(real_metadata):
    assert real_metadata["artifact_schema_version"] == config.ARTIFACT_SCHEMA_VERSION
    assert real_metadata["features"]["columns"] == list(config.FEATURE_COLUMNS)
    assert real_metadata["model"]["selected_k"] == len(real_metadata["clusters"])
    assert len(real_metadata["model"]["candidates"]) == (
        real_metadata["model"]["k_search_range"][1]
        - real_metadata["model"]["k_search_range"][0]
        + 1
    )
    assert set(real_metadata["segment_names"]) == {
        f"C{c['cluster_id']}" for c in real_metadata["clusters"]
    }


def test_real_training_is_reproducible(real_artifacts, real_metadata, real_dataset):
    """Re-fitting on the same CSV with the same seed gives the same labels."""
    _, labels = fit_pipeline(
        real_dataset.features, real_metadata["model"]["selected_k"],
        random_state=real_metadata["model"]["random_state"],
        n_init=real_metadata["model"]["n_init"],
    )
    assert np.array_equal(labels, assign_profiles(real_artifacts, real_dataset.features))
