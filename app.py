"""Customer Segmentation Explorer - Prodigy InfoTech ML Task 02.

Run with:

    python -m streamlit run app.py

The app never trains anything. It reads the artifacts written by
``python -m src.train`` and calls the same inference code as the tests and the
command line, so the numbers on screen and the numbers in the notebook come
from one place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config
from src import visualization as viz
from src.predict import (
    ArtifactError,
    InputValidationError,
    LoadedArtifacts,
    SegmentAssignment,
    assign_profile,
    get_artifact_paths,
    load_artifacts,
)

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

LICENSE_SHORT = config.DATASET_LICENSE.split(" - ")[0]

INCOME_LABEL = config.FEATURE_LABELS[config.INCOME_COLUMN]
SCORE_LABEL = config.FEATURE_LABELS[config.SPENDING_SCORE_COLUMN]


def md(text: str) -> str:
    """Escape ``$`` so the markdown renderer cannot read it as LaTeX math.

    Streamlit renders markdown with KaTeX math switched on, so a bare ``$`` in
    prose - as in the unit "k$" - can be swallowed as the opening delimiter of a
    math span and garble the rest of the line. A CommonMark backslash escape is
    invisible in the rendered output and yields a literal ``$``.

    Only ever wrap text destined for a markdown-rendering element. Widget
    labels, ``help`` text, table headers and plain ``print`` output are not
    markdown, so escaping those would show a stray backslash to the reader.
    """
    return text.replace("$", r"\$")


# ---------------------------------------------------------------------------
# Cached, read-only loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_artifacts(
    model_path: str, metadata_path: str, model_stamp: float, meta_stamp: float
) -> LoadedArtifacts:
    """Load the trusted artifacts once per session. Nothing is ever refitted."""
    del model_stamp, meta_stamp  # only used to invalidate the cache
    return load_artifacts(Path(model_path), Path(metadata_path))


@st.cache_data(show_spinner=False)
def _load_plot_inputs(metadata: dict, stamps: tuple[float, ...]) -> tuple:
    """Rebuild the small tables the figures need from the saved metadata."""
    del stamps  # cache key only
    candidates = pd.DataFrame.from_records(metadata["model"]["candidates"])
    for column in ("k", "inertia", "silhouette", "n_clusters_found", "note"):
        if column in candidates:
            candidates[column] = candidates[column].astype(object)
    summary = pd.DataFrame.from_records(metadata["clusters"])
    centroids = pd.DataFrame(
        {
            "cluster_id": summary["cluster_id"],
            config.INCOME_COLUMN: summary[config.METADATA_CENTROID_INCOME_KEY],
            config.SPENDING_SCORE_COLUMN: summary[config.METADATA_CENTROID_SCORE_KEY],
        }
    )
    return candidates, summary, centroids


@st.cache_data(show_spinner=False)
def _load_footprint(path: str, stamp: float) -> dict:
    del stamp  # cache key only
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_row_level_data(path: str, stamp: float) -> tuple:
    """Optional local-only row level table. Not bundled with a hosted deployment."""
    del stamp  # cache key only
    frame = pd.read_csv(path)
    return frame.loc[:, list(config.FEATURE_COLUMNS)], frame["cluster_id"].to_numpy(int)


def _stamp(path: Path) -> float:
    """File modification time, used to invalidate the caches after a retrain."""
    return path.stat().st_mtime if path.is_file() else -1.0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(metadata: dict) -> None:
    cleaning = metadata["dataset"]["cleaning"]
    excluded = ", ".join(metadata["features"]["excluded_from_model"])
    features = "\n".join(
        f"  - {md(label)}" for label in config.FEATURE_LABELS.values()
    )
    with st.sidebar:
        with st.expander("At a glance", expanded=True):
            st.markdown(
                md(
                    f"- **Customers analysed:** "
                    f"{metadata['features']['training_rows']}\n"
                    f"- **Selected k:** {metadata['model']['selected_k']}\n"
                    f"- **Silhouette** (clustering metric, not accuracy): "
                    f"{metadata['model']['silhouette']:.3f}"
                )
            )
            st.markdown("**Features used**")
            st.markdown(features)
            st.markdown(
                md(f"**Excluded from the model:** {excluded}")
            )

        with st.expander("Dataset"):
            st.markdown(
                md(
                    f"**{metadata['dataset']['title']}**  \n"
                    f"{metadata['dataset']['author']}  \n"
                    f"Licence shown on the source page: {LICENSE_SHORT}  \n"
                    f"Rows used: {cleaning['rows_kept']} of "
                    f"{cleaning['rows_total']}"
                )
            )
            st.markdown(
                f"[Open the dataset on Kaggle]({metadata['dataset']['url']})"
            )
            st.caption(f"File: {metadata['dataset']['local_file']}")
            st.caption(f"SHA-256: {metadata['dataset']['sha256']}")

        with st.expander("Run it locally"):
            st.code(
                "python -m src.train\npython -m streamlit run app.py",
                language="powershell",
            )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def render_intro(metadata: dict) -> None:
    st.markdown(
        "**Prodigy InfoTech Machine Learning Task 02 - customer segmentation using "
        "K-means.** Customers are grouped by how their income and spending score "
        "compare with everyone else in this file."
    )

    cleaning = metadata["dataset"]["cleaning"]
    reasons = cleaning["exclusion_reasons"] or "no invalid rows"
    columns = st.columns(4)
    columns[0].metric("Customers analysed", metadata["features"]["training_rows"])
    columns[1].metric("Selected k", metadata["model"]["selected_k"])
    columns[2].metric(
        "Silhouette (clustering metric)", f"{metadata['model']['silhouette']:.3f}"
    )
    columns[3].metric("Rows excluded while cleaning", cleaning["rows_excluded"], help=reasons)

    st.info(
        md(
            "This dataset holds **customer profiles and an assigned spending score, "
            "not transaction histories**, so it cannot answer what someone bought, "
            "how often, or when they last bought."
        ),
        icon="\u2139\ufe0f",
    )


def render_dataset_details(metadata: dict) -> None:
    """Reference detail: the column table, the cleaning policy, the attribution."""
    dataset = metadata["dataset"]
    cleaning = dataset["cleaning"]
    reasons = cleaning["exclusion_reasons"] or "no invalid rows"

    with st.expander("Dataset details and cleaning policy"):
        st.markdown(
            md(
                f"**{dataset['title']}** by {dataset['author']}, "
                f"from [Kaggle]({dataset['url']})."
            )
        )
        st.markdown(
            md(
                "| Column | Role in this project | Unit |\n"
                "| --- | --- | --- |\n"
                f"| {config.ID_COLUMN} | Identifier only - **not** used by the model "
                "| - |\n"
                f"| {config.GENDER_COLUMN}, {config.AGE_COLUMN} | In the file - "
                "**not** used by the model | - |\n"
                f"| {INCOME_LABEL} | **Clustering feature** | "
                f"{config.FEATURE_UNITS[config.INCOME_COLUMN]} |\n"
                f"| {SCORE_LABEL} | **Clustering feature** | "
                f"{config.FEATURE_UNITS[config.SPENDING_SCORE_COLUMN]} |"
            )
        )

        st.warning(md(dataset["limitation"]), icon="\u26a0\ufe0f")
        st.info(
            md(
                "Because of that limit, this project does **not** analyse transaction "
                "history, purchase frequency or recency, and it does not compute RFM "
                "features. The spending score is one assigned summary number, and "
                "that is all the model uses."
            ),
            icon="\u2139\ufe0f",
        )
        st.caption(dataset["acknowledgement"])

        st.markdown("**Cleaning policy**")
        st.markdown(
            md(
                f"- **Complete cases only.** A row is kept only when both predictors "
                f"are numeric and finite, income is at least {config.INCOME_MIN:g}, "
                f"and the spending score is a whole number from "
                f"{config.SPENDING_SCORE_MIN:g} to {config.SPENDING_SCORE_MAX:g}. "
                "Nothing is imputed, because an invented income or score would be an "
                "invented customer.\n"
                f"- **Reported:** {cleaning['rows_total']} rows in the file, "
                f"{cleaning['rows_kept']} used, {cleaning['rows_excluded']} excluded "
                f"({reasons}).\n"
                f"- **Customers who share income *and* spending score are kept.** "
                f"{cleaning['duplicate_feature_pairs']} customer(s) match someone else "
                "on both numbers. They are different people, so none of them were "
                "removed.\n"
                f"- **Only records identical in *every* column** are treated as "
                f"duplicated records. Count removed: "
                f"{cleaning['exact_duplicate_rows_removed']}.\n"
                f"- **Traceability:** the local export keeps the source row number "
                f"and the dataset's own {config.ID_COLUMN} for every kept row."
            )
        )


def render_clusters_section(
    metadata: dict,
    summary: pd.DataFrame,
    centroids: pd.DataFrame,
    footprint: dict,
) -> None:
    selected_k = int(metadata["model"]["selected_k"])

    st.header("The groups that were found")
    st.pyplot(
        viz.plot_cluster_footprint(footprint, centroids, selected_k), clear_figure=True
    )
    st.caption(
        "Aggregate view: shaded bins hold customer counts and stars mark the cluster "
        "centroids in the original units. A 2D boundary is an algorithmic grouping, "
        "not proof that a real category of customer exists."
    )

    segments_file = config.OUTPUTS_DIR / config.SEGMENTS_FILENAME
    if segments_file.is_file():
        if st.checkbox(
            "Show one dot per customer (local analysis only)",
            value=False,
            help=(
                "This chart reproduces the dataset on the axes, which is why it is "
                "not part of the published app. The aggregate view above is the "
                "default."
            ),
        ):
            features, labels = _load_row_level_data(
                str(segments_file), _stamp(segments_file)
            )
            st.pyplot(
                viz.plot_cluster_scatter(features, labels, centroids, selected_k),
                clear_figure=True,
            )
    else:
        st.caption(
            "Individual customer points are not bundled with this app, so only the "
            "aggregate view is shown."
        )

    left, right = st.columns(2)
    with left:
        st.pyplot(viz.plot_cluster_sizes(summary, selected_k), clear_figure=True)
    with right:
        st.subheader("Group summary")
        st.dataframe(
            summary.rename(
                columns={
                    "cluster_id": "Cluster",
                    "customers": "Customers",
                    "share_of_customers": "Share",
                    config.METADATA_CENTROID_INCOME_KEY: "Centroid income (k$)",
                    config.METADATA_CENTROID_SCORE_KEY: "Centroid score",
                    config.METADATA_MEAN_INCOME_KEY: "Mean income (k$)",
                    config.METADATA_MEAN_SCORE_KEY: "Mean score",
                    "description": "Description",
                    "color": "Colour",
                }
            ).style.format(
                {
                    "Share": "{:.1%}",
                    "Centroid income (k$)": "{:.1f}",
                    "Centroid score": "{:.1f}",
                    "Mean income (k$)": "{:.1f}",
                    "Mean score": "{:.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Cluster ids are arbitrary labels, not a ranking of value. Descriptions are "
        "relative to this dataset's averages and say nothing about age, gender or "
        "loyalty, because those columns were never given to the model."
    )


def render_k_selection_section(candidates: pd.DataFrame, metadata: dict) -> None:
    model = metadata["model"]
    selected_k = int(model["selected_k"])
    with st.expander("How k was chosen"):
        st.markdown(
            f"**The rule used here:** {model['selection_rule']}\n\n"
            f"- Inertia is recorded for k = {model['k_search_range'][0]} to "
            f"{model['k_search_range'][1]} and drawn as the elbow curve.\n"
            "- The silhouette score is only defined for k of 2 or more, so it is the "
            "metric that picks k.\n"
            "- The elbow curve is supporting evidence only. Inertia always falls as k "
            "rises, so it can never choose k on its own."
        )

        left, right = st.columns(2)
        with left:
            st.pyplot(viz.plot_elbow(candidates, selected_k), clear_figure=True)
        with right:
            st.pyplot(viz.plot_silhouette(candidates, selected_k), clear_figure=True)

        st.dataframe(
            candidates.rename(
                columns={
                    "k": "k",
                    "inertia": "Inertia",
                    "silhouette": "Silhouette",
                    "silhouette_valid": "Scored",
                    "n_clusters_found": "Groups found",
                    "note": "Note",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.info(
            f"**k = {selected_k} is a practical choice inside the tested range.** It "
            "is not a discovered truth, and it was not chosen because a tutorial "
            "happened to use the same number.",
            icon="\U0001f3af",
        )
        st.markdown(
            "**What these numbers are, and what they are not**\n\n"
            "- The **silhouette score** compares how far a customer sits from its own "
            "group with how far it sits from the nearest other group. Higher means the "
            "groups are more separated and more internally tight. It is a *clustering "
            "metric*, not a prediction accuracy.\n"
            "- There are **no ground-truth segment labels** in this dataset, so accuracy, "
            "precision and recall do not exist here and are not reported.\n"
            "- Every number is **descriptive and measured on the dataset the model was "
            "fitted on**. It is not held-out supervised validation and it does not "
            "promise the groups will hold for new customers.\n"
            "- **Cluster numbers are arbitrary identifiers.** Renumbering them would "
            "describe exactly the same grouping."
        )


def render_how_it_works() -> None:
    with st.expander("How K-means works"):
        st.markdown(
            "1. **Standardise the columns.** StandardScaler gives both features "
            "comparable variance so their numerical scales do not disproportionately "
            "influence distances. It subtracts each feature's mean and divides by its "
            "standard deviation. Whether an unscaled feature dominates depends on its "
            "actual numerical spread.\n"
            "2. **Choose k.** K-means must be told how many groups to look for. This "
            "project tests several values and keeps the one with the best silhouette "
            "score.\n"
            "3. **Place k centres.** K-means starts from k random points, called seeds, "
            "and treats them as group centres.\n"
            "4. **Assign.** Every customer is attached to the nearest centre, where "
            "'nearest' means smallest distance in the standardised two-number space.\n"
            "5. **Move the centres.** Each centre moves to the average of its members.\n"
            "6. **Repeat steps 4 and 5** until the centres stop moving. That is the "
            "converged result.\n"
            "7. **Describe each group** by comparing its final centre with the dataset "
            "average, for example 'higher income, lower spending score'."
        )
        st.markdown(
            f"Two settings make the result repeatable:\n\n"
            f"- random_state={config.RANDOM_STATE} fixes the random seed, so the same "
            f"data produces the same groups.\n"
            f"- n_init={config.N_INIT} starts K-means {config.N_INIT} times from "
            "different seeds and keeps the best run, which reduces the chance of "
            "landing on a poor local solution."
        )
        st.caption(
            "K-means assumes the groups are roughly round and of similar size in the "
            "feature space. That is a modelling assumption about these two numbers, "
            "not a fact about how customers behave."
        )


def _distance_table(artifacts: LoadedArtifacts, result: SegmentAssignment) -> pd.DataFrame:
    """One row per learned cluster: its saved centroid and the profile's distance.

    Centroids and descriptions are read from the fitted metadata and joined on
    ``cluster_id``, so each column holds the quantity its header names. Distances
    come from the same standardised-space calculation as the assigned result.
    """
    centroids = artifacts.cluster_summary.set_index(config.METADATA_CLUSTER_ID_KEY)
    names = artifacts.metadata["segment_names"]

    rows = []
    for cluster_id, distance in sorted(result.distances_standardized.items()):
        if cluster_id not in centroids.index:
            raise KeyError(f"C{cluster_id} is missing from the fitted cluster summary")
        saved = centroids.loc[cluster_id]
        rows.append(
            {
                "Cluster": f"C{cluster_id}",
                "Description": names[f"C{cluster_id}"],
                "Centroid income (k$)": float(
                    saved[config.METADATA_CENTROID_INCOME_KEY]
                ),
                "Centroid spending score": float(
                    saved[config.METADATA_CENTROID_SCORE_KEY]
                ),
                "Distance (standardised units)": float(distance),
            }
        )
    return pd.DataFrame(rows)


def render_assignment_form(metadata: dict, artifacts: LoadedArtifacts) -> None:
    st.header("Assign a customer profile")
    st.markdown(
        "Enter an income and a spending score that already exist, and the app will "
        "match them to the closest group the model learned. **This assigns an "
        "existing spending score to a group - it does not predict future spending.**"
    )

    ranges = metadata["features"]["training_ranges"]
    income_stats = ranges[config.INCOME_COLUMN]
    score_stats = ranges[config.SPENDING_SCORE_COLUMN]

    with st.form("assign_profile_form", clear_on_submit=False):
        income = st.number_input(
            INCOME_LABEL,
            min_value=0.0,
            max_value=float(income_stats["max"]) * 3,
            value=round((income_stats["min"] + income_stats["max"]) / 2, 1),
            step=1.0,
            format="%.1f",
            help=(
                "Thousands, exactly as the source column is labelled. This dataset "
                f"spans {income_stats['min']:g} to {income_stats['max']:g} in these "
                "units."
            ),
        )
        score = st.number_input(
            SCORE_LABEL,
            min_value=int(config.SPENDING_SCORE_MIN),
            max_value=int(config.SPENDING_SCORE_MAX),
            value=int(
                min(
                    max(round(score_stats["mean"]), int(config.SPENDING_SCORE_MIN)),
                    int(config.SPENDING_SCORE_MAX),
                )
            ),
            step=1,
            help="An assigned score from 1 to 100, on the same scale as the dataset.",
        )
        submitted = st.form_submit_button("Assign to nearest cluster", type="primary")

    if not submitted:
        st.caption(
            "The validation rules live in src/predict.py, not only in these "
            "widgets, so the app and the command line accept and reject exactly the "
            "same values."
        )
        return

    try:
        result = assign_profile(artifacts, income, score)
    except InputValidationError as exc:
        st.error(f"That profile cannot be used: {exc}", icon="🚫")
        return

    centroid_income = result.centroid[config.INCOME_COLUMN]
    centroid_score = result.centroid[config.SPENDING_SCORE_COLUMN]

    st.success(f"Assigned to cluster **C{result.cluster_id}**", icon="✅")
    st.markdown(f"**Description:** {result.description} *(relative to this dataset's averages)*")
    st.markdown(
        md(
            f"**Centroid of C{result.cluster_id}:** {centroid_income:,.1f} k$ annual "
            f"income, spending score {centroid_score:,.1f}"
        )
    )
    st.markdown(
        "**Distance to this centroid:** "
        f"{result.nearest_distance_standardized:.3f} standardised units"
    )
    st.caption(
        "That distance is measured in the standardised feature space, where each "
        "column has mean 0 and standard deviation 1. It is a geometric distance - not "
        "a confidence score, and not a probability of belonging to the group."
    )

    st.markdown("**Distance to every learned centroid** (standardised units)")
    st.dataframe(
        _distance_table(artifacts, result),
        width="stretch",
        hide_index=True,
    )

    for note in result.out_of_range_notes:
        st.warning(note, icon="⚠️")

    total = int(artifacts.metadata["features"]["training_rows"])
    size = artifacts.cluster_summary.loc[
        artifacts.cluster_summary["cluster_id"] == result.cluster_id, "customers"
    ].iloc[0]
    st.caption(
        f"Cluster C{result.cluster_id} holds {int(size)} of the {total} customers in "
        "the fitted model, about "
        f"{int(size) / total:.1%} of the dataset."
    )


def render_limits_section(metadata: dict) -> None:
    st.header("Assumptions and limitations")
    st.markdown("\n".join(f"- {note}" for note in metadata["interpretation_notes"]))
    st.markdown(
        md(
            "- **Units are the dataset's own.** Annual income stays in k$ "
            "(thousands). The file is not tied to a currency for any particular "
            "country, so this project does not describe it as INR or as current "
            "Indian customers.\n"
            "- **The clusters are not personas.** They are positions in a two-number "
            "space. The model never saw age or gender, so it cannot claim anything "
            "about them.\n"
            "- **A profile outside the training range is still assigned** to the "
            "nearest centre, with a warning, because the model has no better answer.\n"
            f"- **Serving the app never retrains.** The model is read from "
            f"{config.MODEL_FILENAME}; the raw CSV is not needed at runtime and is "
            "not bundled."
        )
    )
    st.caption(
        "**On the source terms.** The dataset page lists the licence as "
        f'"{LICENSE_SHORT}" and does not state a standard open licence. That is an '
        "absence of a clear grant, not an explicit prohibition, and it was not "
        "possible to confirm permission to redistribute the file from the page "
        "itself. Keeping the raw CSV, the row-level export and the per-customer "
        "chart out of the published bundle is a precaution taken because the terms "
        "are unclear. Aggregating the counts reduces what the chart reveals, but "
        "it does not by itself settle the licensing question and it is not a "
        "guarantee of anonymity."
    )


def render_technical_details(metadata: dict) -> None:
    with st.expander("Technical details"):
        dataset = metadata["dataset"]
        st.markdown(
            md(
                f"- **Source file:** {dataset['local_file']}\n"
                f"- **SHA-256:** {dataset['sha256']}\n"
                f"- **Trained:** {metadata['trained_at_utc']}\n"
                f"- **random_state:** {metadata['model']['random_state']}, "
                f"**n_init:** {metadata['model']['n_init']}\n"
                f"- **k search range:** {metadata['model']['k_search_range'][0]} to "
                f"{metadata['model']['k_search_range'][1]}"
            )
        )
        st.markdown(
            md(
                "**Feature units as recorded at training time**\n\n"
                + "\n".join(
                    f"- {label}: {config.FEATURE_UNITS[column]}"
                    for column, label in config.FEATURE_LABELS.items()
                )
            )
        )
        st.caption(
            "The app reads these artifacts; it never refits anything. Re-run "
            "python -m src.train to rebuild them."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.title(config.APP_TITLE)
    st.caption(
        f"{config.PROJECT_TITLE} - unsupervised learning, there are no labels"
    )

    model_path, metadata_path = get_artifact_paths()
    stamps = (_stamp(model_path), _stamp(metadata_path))

    try:
        artifacts = _load_artifacts(
            str(model_path), str(metadata_path), stamps[0], stamps[1]
        )
    except ArtifactError as exc:
        st.error("The model is not ready yet.", icon="\U0001f6a7")
        st.markdown(str(exc))
        st.code("python -m src.train", language="powershell")
        st.stop()

    metadata = artifacts.metadata
    footprint_path = config.MODELS_DIR / config.FOOTPRINT_FILENAME
    if not footprint_path.is_file():
        st.error(
            "The aggregate cluster footprint is missing. Re-create the artifacts "
            "with: python -m src.train"
        )
        st.stop()

    footprint = _load_footprint(str(footprint_path), _stamp(footprint_path))
    candidates, summary, centroids = _load_plot_inputs(metadata, stamps)

    render_sidebar(metadata)

    # Main workflow first: what the groups are, then the interactive form.
    render_intro(metadata)
    st.divider()
    render_clusters_section(metadata, summary, centroids, footprint)
    st.divider()
    render_assignment_form(metadata, artifacts)
    st.divider()

    # Supporting explanation, folded away by default.
    render_k_selection_section(candidates, metadata)
    render_how_it_works()
    render_dataset_details(metadata)
    render_technical_details(metadata)
    st.divider()

    render_limits_section(metadata)

    st.caption(metadata["dataset"]["acknowledgement"])


if __name__ == "__main__":
    main()
