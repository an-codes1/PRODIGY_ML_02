"""Every figure used by the project.

Two scatter views exist on purpose:

* :func:`plot_cluster_scatter` draws one dot per customer. It is the clearest
  teaching view, but it *is* the dataset drawn on the axes, so it stays local
  and is not published.
* :func:`plot_cluster_footprint` draws only binned counts plus the centroids.
  No individual customer is recoverable from it, so this is the view used by the
  Streamlit app and the one that is safe to publish.

A 2D cluster boundary is an algorithmic grouping produced by K-means, not proof
that a real "type" of customer exists.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import matplotlib

# Headless, CPU-only rendering. An already-inline backend (a notebook that ran
# ``%matplotlib inline``) is left alone so the figures display there; anything
# else is forced to Agg, which never needs a display or a GUI toolkit.
_HEADLESS_BACKENDS = {
    "agg",
    "module://matplotlib_inline.backend_inline",
    "nbagg",
    "notebook",
    "ipympl",
    "module://ipympl.backend_nbagg",
    "widget",
}
if not os.environ.get("MPLBACKEND") and matplotlib.get_backend().lower() not in _HEADLESS_BACKENDS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import config

INCOME = config.INCOME_COLUMN
SCORE = config.SPENDING_SCORE_COLUMN

AXIS_X_LABEL = "Annual income (k$) - thousands, as labelled by the dataset"
AXIS_Y_LABEL = "Spending score (1-100) - assigned by the dataset"

_CAPTION_SCOPE = (
    "Groups come from K-means on these two columns only, so a boundary is an "
    "algorithmic grouping and not proof of a real customer category."
)


def _finish(fig, ax, caption: str | None = None) -> None:
    """Grid, then a caption strip reserved at the bottom, then layout.

    Reserving space with ``rect`` keeps the caption from being clipped and stops
    ``tight_layout`` from fighting with the long axis labels.
    """
    ax.grid(alpha=0.25, linewidth=0.6)
    if caption:
        fig.text(
            0.02, 0.015,
            textwrap.fill(caption, width=118),
            fontsize=8.5, color="#444444", ha="left", va="bottom",
        )
        fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0))
    else:
        fig.tight_layout()


def _save(fig, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _colour(cluster_id: int) -> str:
    return config.CLUSTER_COLORS[int(cluster_id) % len(config.CLUSTER_COLORS)]


def plot_elbow(candidates: pd.DataFrame, selected_k: int, path: Path | None = None):
    """Inertia (within-cluster sum of squares) against k."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    valid = candidates.dropna(subset=["inertia"])
    ax.plot(valid["k"], valid["inertia"], marker="o", color=_colour(0))
    if selected_k is not None and selected_k in set(valid["k"]):
        point = valid.loc[valid["k"] == selected_k].iloc[0]
        ax.scatter(
            [point["k"]], [point["inertia"]], s=140, zorder=3,
            facecolor="none", edgecolor=_colour(3), linewidth=2.2,
        )
        ax.annotate(
            f"selected k = {selected_k}",
            (point["k"], point["inertia"]),
            textcoords="offset points", xytext=(8, 10), fontsize=10,
        )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow curve - lower is tighter, and it always falls", fontsize=12)
    ax.set_xticks(valid["k"].tolist())
    _finish(
        fig, ax,
        "Supporting evidence only. Inertia keeps dropping as k rises, so it cannot "
        "pick k on its own. " + _CAPTION_SCOPE,
    )
    _save(fig, path)
    return fig


def plot_silhouette(candidates: pd.DataFrame, selected_k: int, path: Path | None = None):
    """Silhouette score against k, with the selected k highlighted."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    scored = candidates.dropna(subset=["silhouette"])
    ax.plot(scored["k"], scored["silhouette"], marker="o", color=_colour(1))
    if selected_k is not None and selected_k in set(scored["k"]):
        point = scored.loc[scored["k"] == selected_k].iloc[0]
        ax.scatter(
            [point["k"]], [point["silhouette"]], s=140, zorder=3,
            facecolor="none", edgecolor=_colour(3), linewidth=2.2,
        )
        ax.annotate(
            f"selected k = {selected_k}  (silhouette {point['silhouette']:.3f})",
            (point["k"], point["silhouette"]),
            textcoords="offset points", xytext=(8, 10), fontsize=10,
        )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (clustering metric, higher is better)")
    ax.set_title("Silhouette score by k - the rule used to pick k here", fontsize=12)
    ax.set_xticks(candidates["k"].tolist())
    ax.set_ylim(0, 1)
    _finish(
        fig, ax,
        "Silhouette measures how separated and how internally tight the groups are. "
        "It is not a prediction score; there are no true segment labels to score "
        "against. " + _CAPTION_SCOPE,
    )
    _save(fig, path)
    return fig


def build_footprint(
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    income_bins: int = config.FOOTPRINT_INCOME_BINS,
    spending_bins: int = config.FOOTPRINT_SPENDING_BINS,
) -> dict:
    """Bin the two predictors into a small count grid.

    This is the aggregate form of the scatter plot that gets saved next to the
    model, so the hosted app can draw a cluster view without the raw CSV.
    """
    income = features[INCOME].to_numpy(dtype=float)
    score = features[SCORE].to_numpy(dtype=float)
    counts, income_edges, score_edges = np.histogram2d(
        income, score, bins=[income_bins, spending_bins]
    )
    return {
        "income_bin_edges": [float(v) for v in income_edges],
        "spending_score_bin_edges": [float(v) for v in score_edges],
        "counts": [[int(v) for v in row] for row in counts],
        "note": (
            "Binned customer counts only. The grid is an aggregate of the cleaned "
            "rows; no individual customer record can be recovered from it."
        ),
    }


def plot_cluster_footprint(
    footprint: dict,
    centroids: pd.DataFrame,
    selected_k: int,
    path: Path | None = None,
):
    """Aggregate cluster view: binned counts, then the centroids on top.

    This is the default cluster plot in the app. It shows where the groups sit
    without republishing individual customer coordinates.
    """
    income_edges = np.asarray(footprint["income_bin_edges"], dtype=float)
    score_edges = np.asarray(footprint["spending_score_bin_edges"], dtype=float)
    counts = np.asarray(footprint["counts"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    mesh = ax.pcolormesh(
        income_edges, score_edges,
        np.ma.masked_less_equal(np.log1p(counts), 0).T,
        cmap="viridis", shading="auto",
    )
    colour_bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colour_bar.set_label("log(1 + customers in bin) - aggregate counts", fontsize=9)

    handles = []
    for _, row in centroids.sort_values("cluster_id").iterrows():
        cluster_id = int(row["cluster_id"])
        colour = _colour(cluster_id)
        ax.scatter(
            [row[INCOME]], [row[SCORE]], marker="*", s=320, c=colour,
            edgecolor="black", linewidth=1.1, zorder=4,
        )
        ax.annotate(
            f"C{cluster_id}", (row[INCOME], row[SCORE]),
            textcoords="offset points", xytext=(9, 7),
            fontsize=10, fontweight="bold", color=colour, zorder=5,
        )
        handles.append(
            Line2D(
                [0], [0], marker="*", linestyle="none", markersize=14,
                markerfacecolor=colour, markeredgecolor="black",
                label=f"C{cluster_id} centroid",
            )
        )

    ax.set_xlabel(AXIS_X_LABEL)
    ax.set_ylabel(AXIS_Y_LABEL)
    ax.set_title(
        f"Where the groups sit in the data - aggregate view, k = {selected_k}", fontsize=12
    )
    ax.legend(
        handles=handles, title="Centroids (original units)", fontsize=9,
        title_fontsize=9, loc="best", framealpha=0.9,
    )
    _finish(
        fig, ax,
        "Shaded bins hold aggregated customer counts, not individual records. Stars "
        "mark cluster centroids in k$ and score units. " + _CAPTION_SCOPE,
    )
    _save(fig, path)
    return fig


def plot_cluster_scatter(
    features: pd.DataFrame,
    labels: np.ndarray,
    centroids: pd.DataFrame,
    selected_k: int,
    path: Path | None = None,
):
    """One dot per customer, coloured by cluster, with the centroids marked.

    Local teaching view. It plots the dataset itself, so it is not published.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    income = features[INCOME].to_numpy(dtype=float)
    score = features[SCORE].to_numpy(dtype=float)

    for cluster_id in sorted({int(v) for v in labels}):
        member = labels == cluster_id
        ax.scatter(
            income[member], score[member], s=42, alpha=0.75, color=_colour(cluster_id),
            edgecolor="white", linewidth=0.5, label=f"C{cluster_id} ({int(member.sum())})",
        )

    ordered = centroids.sort_values("cluster_id")
    ax.scatter(
        ordered[INCOME], ordered[SCORE], marker="*", s=340,
        c=[_colour(int(cid)) for cid in ordered["cluster_id"]],
        edgecolor="black", linewidth=1.1, zorder=4, label="Centroid",
    )

    ax.set_xlabel(AXIS_X_LABEL)
    ax.set_ylabel(AXIS_Y_LABEL)
    ax.set_title(f"One dot per customer, grouped by K-means (k = {selected_k})", fontsize=12)
    ax.legend(title="Cluster (customers)", fontsize=9, title_fontsize=9, loc="best",
              framealpha=0.9)
    _finish(
        fig, ax,
        "Local view only: this chart shows the dataset itself and is not published. "
        + _CAPTION_SCOPE,
    )
    _save(fig, path)
    return fig


def plot_cluster_sizes(summary: pd.DataFrame, selected_k: int, path: Path | None = None):
    """How many customers landed in each group."""
    ordered = summary.sort_values("cluster_id")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [f"C{int(cid)}" for cid in ordered["cluster_id"]]
    colours = [_colour(cid) for cid in ordered["cluster_id"]]
    bars = ax.bar(labels, ordered["customers"], color=colours, edgecolor="black",
                  linewidth=0.6)
    for bar, value in zip(bars, ordered["customers"]):
        ax.annotate(
            str(int(value)), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9,
        )
    ax.set_xlabel("Cluster id (an arbitrary label, not a ranking)")
    ax.set_ylabel("Customers in the group")
    ax.set_title(f"Group sizes for k = {selected_k}", fontsize=12)
    ax.set_ylim(0, float(ordered["customers"].max()) * 1.15)
    _finish(
        fig, ax,
        "Cluster numbers are arbitrary identifiers. Renumbering them would not change "
        "the grouping.",
    )
    _save(fig, path)
    return fig
