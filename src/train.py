"""Training entry point: ``python -m src.train``.

Steps
-----
1. Read and validate the CSV (:mod:`src.data`).
2. Standardise the two predictors, then fit K-means for every candidate k.
3. Record inertia for k=1..k_max and the silhouette score for k=2..k_max.
4. Select the k with the highest measured silhouette score.
5. Fit the final scaler + K-means pipeline and describe each centroid.
6. Write the trusted artifacts and the result files.

Run ``python -m src.train --help`` for the options.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .clustering import (
    KMEANS_STEP_NAME,
    build_cluster_summary,
    centroids_in_original_units,
    evaluate_candidate_ks,
    fit_pipeline,
    select_k,
)
from .data import DataValidationError, clean_dataset, dataset_fingerprint, load_dataset, validate_clustering_input
from .visualization import build_footprint, plot_cluster_scatter, plot_cluster_sizes, plot_elbow, plot_silhouette, plot_cluster_footprint

_TRACKED_DEPENDENCIES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "joblib",
    "streamlit",
)


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in _TRACKED_DEPENDENCIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover
            versions[name] = "not installed"
    return versions


def _json_safe(value):
    """Convert numpy scalars to plain Python and NaN to ``None``.

    ``json.dumps`` refuses numpy types and would happily write invalid ``NaN``
    tokens, so every value that reaches the metadata file goes through here.
    """
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    return value


def _json_safe_records(frame: pd.DataFrame) -> list[dict]:
    return [
        {key: _json_safe(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _segment_name_mapping(summary: pd.DataFrame) -> dict[str, str]:
    """The human-readable name kept next to every cluster id.

    The app, the CSV export and the metadata all use this mapping, so a cluster
    id always means the same thing.
    """
    return {f"C{int(getattr(row, config.METADATA_CLUSTER_ID_KEY))}": getattr(row, config.METADATA_DESCRIPTION_KEY) for row in summary.itertuples()}


@dataclass
class TrainingResult:
    pipeline: object
    metadata: dict
    candidates: pd.DataFrame
    summary: pd.DataFrame
    segments: pd.DataFrame
    selected_k: int
    written_files: list[Path]


def run_training(
    csv_path: Path | str | None = None,
    *,
    k_max: int = config.K_MAX,
    random_state: int = config.RANDOM_STATE,
    n_init: int = config.N_INIT,
    models_dir: Path | None = None,
    outputs_dir: Path | None = None,
    write_figures: bool = True,
) -> TrainingResult:
    """Fit the model and write every artifact. Returns the results in memory."""
    data_file = Path(csv_path) if csv_path is not None else config.DATA_FILE
    model_dir = Path(models_dir) if models_dir is not None else config.MODELS_DIR
    output_dir = Path(outputs_dir) if outputs_dir is not None else config.OUTPUTS_DIR
    figure_dir = output_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data -------------------------------------------------------------
    raw = load_dataset(data_file)
    dataset = clean_dataset(raw)
    validate_clustering_input(dataset)
    features = dataset.features

    # 2. Candidate k ------------------------------------------------------
    candidates = evaluate_candidate_ks(
        features, k_max=k_max, random_state=random_state, n_init=n_init
    )
    selected_k = select_k(candidates)

    # 3. Final fit --------------------------------------------------------
    pipeline, labels = fit_pipeline(
        features, selected_k, random_state=random_state, n_init=n_init
    )
    centroids = centroids_in_original_units(pipeline)
    summary = build_cluster_summary(features, labels, centroids)
    footprint = build_footprint(features, labels)

    scored = candidates.loc[candidates["k"] == selected_k].iloc[0]
    inertia = float(pipeline.named_steps[KMEANS_STEP_NAME].inertia_)

    # 4. Row-level assignments (local file, not published) ----------------
    segments = dataset.frame.copy()
    segments["cluster_id"] = labels
    segments["segment_name"] = [f"C{int(cid)}" for cid in labels]
    segments["segment_description"] = [
        str(summary.loc[summary[config.METADATA_CLUSTER_ID_KEY] == int(cid), config.METADATA_DESCRIPTION_KEY].iloc[0])
        for cid in labels
    ]
    ordered_columns = [
        config.SOURCE_ROW_COLUMN,
        config.ID_COLUMN,
        *config.OPTIONAL_DEMOGRAPHIC_COLUMNS,
        *config.FEATURE_COLUMNS,
        "cluster_id",
        "segment_name",
        "segment_description",
    ]
    segments = segments.loc[:, [c for c in ordered_columns if c in segments.columns]]

    # 5. Metadata ---------------------------------------------------------
    metadata = {
        "artifact_schema_version": config.ARTIFACT_SCHEMA_VERSION,
        "project_title": config.PROJECT_TITLE,
        "app_title": config.APP_TITLE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "Prodigy InfoTech Machine Learning Task 02",
        "objective": (
            "Group retail customers by spending behaviour using K-means clustering."
        ),
        "dataset": {
            "title": config.DATASET_TITLE,
            "url": config.DATASET_URL,
            "author": config.DATASET_AUTHOR,
            "license": config.DATASET_LICENSE,
            "acknowledgement": config.DATASET_ACKNOWLEDGEMENT,
            "limitation": config.DATASET_LIMITATION,
            "local_file": str(data_file.name),
            "sha256": dataset_fingerprint(data_file),
            "cleaning": dataset.report.to_dict(),
        },
        "features": {
            "columns": list(config.FEATURE_COLUMNS),
            "units": dict(config.FEATURE_UNITS),
            "labels": dict(config.FEATURE_LABELS),
            "excluded_from_model": list(config.EXCLUDED_FROM_MODEL),
            "exclusion_reason": (
                "CustomerID is an identifier, not behaviour. Gender and Age are "
                "demographics; leaving them out keeps the groups defined purely by "
                "income and spending score."
            ),
            "training_rows": int(len(features)),
            "training_ranges": {
                column: {
                    "min": float(features[column].min()),
                    "max": float(features[column].max()),
                    "mean": float(features[column].mean()),
                }
                for column in config.FEATURE_COLUMNS
            },
        },
        "model": {
            "algorithm": "KMeans",
            "library": "scikit-learn",
            "pipeline_steps": ["StandardScaler", "KMeans"],
            "random_state": random_state,
            "n_init": n_init,
            "selected_k": selected_k,
            "k_search_range": [config.K_MIN, int(k_max)],
            "selection_rule": (
                "Highest measured silhouette score among valid candidates k=2.."
                f"{int(k_max)} in the standardised feature space; exact ties go to the "
                "smaller k."
            ),
            "inertia": inertia,
            "silhouette": float(scored["silhouette"]),
            "silhouette_meaning": (
                "Clustering metric between -1 and 1, not a prediction accuracy."
            ),
            "candidates": _json_safe_records(candidates),
        },
        "clusters": [
            {
                key: (
                    round(value, 4) if isinstance(value, float) else value
                )
                for key, value in row.items()
            }
            for row in summary.to_dict(orient="records")
        ],
        "segment_names": _segment_name_mapping(summary),
        "cluster_colors": {
            f"C{int(row[config.METADATA_CLUSTER_ID_KEY])}": row[config.METADATA_COLOR_KEY] for row in summary.to_dict("records")
        },
        "interpretation_notes": [
            "k was chosen inside the tested range "
            f"1..{int(k_max)}; it is a practical choice, not a discovered truth.",
            "The elbow curve is supporting evidence, not the selection rule.",
            "There are no ground-truth segment labels, so no accuracy score exists.",
            "All metrics are descriptive and measured on the fitted dataset. This is "
            "not held-out supervised validation.",
            "Cluster ids are arbitrary identifiers, not a ranking of value.",
            "Centroid descriptions are relative to this dataset's means and make no "
            "claim about age, gender, loyalty or purchase history.",
            "A 2D cluster boundary is an algorithmic grouping, not proof of a real "
            "customer category.",
        ],
        "dependencies": _dependency_versions(),
        "deployment": {
            "row_level_data": (
                "Not published. The Kaggle dataset page lists its licence as "
                "'Other (specified in description)', so the raw CSV and any chart that "
                "reproduces individual customer coordinates are kept local. The app "
                "draws an aggregate binned view from segment_footprint.json."
            ),
            "published_artifacts": [
                config.MODEL_FILENAME,
                config.METADATA_FILENAME,
                config.FOOTPRINT_FILENAME,
            ],
            "training_at_startup": False,
        },
    }

    # 6. Write artifacts --------------------------------------------------
    written: list[Path] = []

    model_file = model_dir / config.MODEL_FILENAME
    joblib.dump(pipeline, model_file)
    written.append(model_file)

    metadata_file = model_dir / config.METADATA_FILENAME
    metadata_file.write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    written.append(metadata_file)

    footprint_file = model_dir / config.FOOTPRINT_FILENAME
    footprint_file.write_text(
        json.dumps(
            {
                "selected_k": selected_k,
                "n_customers": int(len(features)),
                "source": f"{data_file.name} (sha256 {metadata['dataset']['sha256'][:12]}...)",
                **footprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written.append(footprint_file)

    segments_file = output_dir / config.SEGMENTS_FILENAME
    segments.to_csv(segments_file, index=False)
    written.append(segments_file)

    summary_file = output_dir / config.SUMMARY_FILENAME
    summary.round(4).to_csv(summary_file, index=False)
    written.append(summary_file)

    metrics_file = output_dir / config.K_METRICS_FILENAME
    candidates.to_csv(metrics_file, index=False)
    written.append(metrics_file)

    if write_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_paths = {
            "elbow_curve": figure_dir / "elbow_curve.png",
            "silhouette_by_k": figure_dir / "silhouette_by_k.png",
            "cluster_footprint": figure_dir / "cluster_footprint.png",
            "cluster_scatter_local_only": figure_dir / "cluster_scatter_local_only.png",
            "cluster_sizes": figure_dir / "cluster_sizes.png",
        }
        plot_elbow(candidates, selected_k, figure_paths["elbow_curve"])
        plot_silhouette(candidates, selected_k, figure_paths["silhouette_by_k"])
        plot_cluster_footprint(
            footprint, centroids, selected_k, figure_paths["cluster_footprint"]
        )
        plot_cluster_scatter(
            features, labels, centroids, selected_k,
            figure_paths["cluster_scatter_local_only"],
        )
        plot_cluster_sizes(summary, selected_k, figure_paths["cluster_sizes"])
        written.extend(figure_paths.values())

    return TrainingResult(
        pipeline=pipeline,
        metadata=metadata,
        candidates=candidates,
        summary=summary,
        segments=segments,
        selected_k=selected_k,
        written_files=written,
    )


def format_report(result: TrainingResult) -> str:
    """A short, plain-text summary for the terminal."""
    metadata = result.metadata
    cleaning = metadata["dataset"]["cleaning"]
    lines = [
        "",
        "=" * 72,
        f"{config.PROJECT_TITLE}",
        "=" * 72,
        f"Dataset          : {metadata['dataset']['title']}",
        f"Source           : {metadata['dataset']['url']}",
        f"Licence on source: {metadata['dataset']['license']}",
        f"SHA-256          : {metadata['dataset']['sha256']}",
        f"Rows in file     : {cleaning['rows_total']}",
        f"Rows used        : {cleaning['rows_kept']}",
        f"Rows excluded    : {cleaning['rows_excluded']} "
        f"({cleaning['exclusion_reasons'] or 'no invalid rows'})",
        f"Exact duplicates removed : {cleaning['exact_duplicate_rows_removed']}",
        f"Same income+score pairs kept : {cleaning['duplicate_feature_pairs']}",
        f"Features         : {', '.join(config.FEATURE_COLUMNS)}",
        f"Excluded columns : {', '.join(config.EXCLUDED_FROM_MODEL)}",
        "",
        "Candidate k scores (standardised feature space):",
    ]
    for row in result.candidates.to_dict(orient="records"):
        silhouette = (
            f"{row['silhouette']:.4f}"
            if row["silhouette"] is not None and not pd.isna(row["silhouette"])
            else "  n/a"
        )
        inertia = (
            f"{row['inertia']:.2f}"
            if row["inertia"] is not None and not pd.isna(row["inertia"])
            else "n/a"
        )
        note = f"  <- {row['note']}" if row["note"] else ""
        lines.append(
            f"  k={row['k']:>2}  inertia={inertia:>9}  silhouette={silhouette}{note}"
        )
    lines += [
        "",
        f"Selected k       : {result.selected_k} "
        f"(silhouette {metadata['model']['silhouette']:.4f}, "
        f"inertia {metadata['model']['inertia']:.2f})",
        "",
        "Clusters:",
    ]
    for row in result.summary.to_dict(orient="records"):
        centroid_income = row[config.METADATA_CENTROID_INCOME_KEY]
        centroid_score = row[config.METADATA_CENTROID_SCORE_KEY]
        lines.append(
            f"  C{int(row[config.METADATA_CLUSTER_ID_KEY])}  "
            f"n={int(row[config.METADATA_CUSTOMERS_KEY]):>3}  "
            f"centroid income={centroid_income:>7.2f} k$  "
            f"centroid score={centroid_score:>6.2f}  "
            f"({row[config.METADATA_DESCRIPTION_KEY]})"
        )
    lines += [
        "",
        "Artifacts written:",
    ]
    lines += [f"  - {path}" for path in result.written_files]
    lines += [
        "",
        "Reminder: these are descriptive clustering metrics measured on the fitted",
        "dataset. There are no true segment labels, so there is no accuracy score.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Train the K-means customer segmentation model and write artifacts.",
    )
    parser.add_argument(
        "--data", type=Path, default=config.DATA_FILE,
        help=f"Path to Mall_Customers.csv (default: {config.DATA_FILE})",
    )
    parser.add_argument(
        "--k-max", type=int, default=config.K_MAX,
        help=f"Largest candidate k to test (default: {config.K_MAX})",
    )
    parser.add_argument(
        "--random-state", type=int, default=config.RANDOM_STATE,
        help=f"K-means random_state (default: {config.RANDOM_STATE})",
    )
    parser.add_argument(
        "--n-init", type=int, default=config.N_INIT,
        help=f"K-means n_init restarts (default: {config.N_INIT})",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip writing PNG figures (useful on a quick re-run)",
    )
    args = parser.parse_args(argv)

    try:
        result = run_training(
            args.data,
            k_max=args.k_max,
            random_state=args.random_state,
            n_init=args.n_init,
            write_figures=not args.no_figures,
        )
    except (FileNotFoundError, DataValidationError) as exc:
        print(f"\nTraining could not start.\n\n{exc}\n", file=sys.stderr)
        return 1

    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
