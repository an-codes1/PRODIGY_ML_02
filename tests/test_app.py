"""Streamlit app checks using Streamlit's own AppTest runner.

These run the real ``app.py`` script - the same file a browser loads - so they
catch broken widgets, missing sections and unhandled exceptions that a plain
unit test would miss. They are not a substitute for opening the app in a
browser; see the README for the manual visual check.
"""

from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from src import config
from app import md
from tests.conftest import APP_FILE

DEFAULT_TIMEOUT = 120


@pytest.fixture()
def app_with_temp_artifacts(monkeypatch, trained_dirs):
    """Point the app at a freshly trained model in a temporary folder."""
    models_dir, outputs_dir = trained_dirs
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "OUTPUTS_DIR", outputs_dir)
    return AppTest.from_file(str(APP_FILE), default_timeout=DEFAULT_TIMEOUT).run()


def _texts(app) -> str:
    """Every piece of visible text the app rendered, in one string.

    Expander labels are included, because several sections are folded away by
    default and their headings are still part of what a reader sees.
    """
    parts: list[str] = [str(item.label) for item in app.expander]
    for name in (
        "title", "header", "subheader", "markdown", "caption",
        "info", "warning", "success", "error", "text", "code",
    ):
        for element in getattr(app, name):
            value = element.value
            parts.append(value if isinstance(value, str) else str(value))
    return "\n".join(parts)


def _chart_count(app) -> int:
    """``st.pyplot`` figures appear in the element tree as images."""
    return len(app.get("image"))


# ---------------------------------------------------------------------------
# Main sections render
# ---------------------------------------------------------------------------
def test_app_runs_without_exceptions(app_with_temp_artifacts):
    app = app_with_temp_artifacts
    assert not app.exception, [str(e.value) for e in app.exception]


def test_app_shows_the_title_and_the_task_description(app_with_temp_artifacts):
    assert app_with_temp_artifacts.title[0].value == config.APP_TITLE
    text = _texts(app_with_temp_artifacts)
    assert "Task 02" in text
    assert "K-means" in text


def test_app_shows_the_dataset_limitation_and_cleaning_policy(app_with_temp_artifacts):
    warnings = " ".join(item.value for item in app_with_temp_artifacts.warning)
    text = _texts(app_with_temp_artifacts)
    assert "no dated transactions" in warnings or "no dated transactions" in text
    assert "Complete cases only" in text
    assert "spending score" in text.lower()


def test_app_shows_the_key_numbers(app_with_temp_artifacts):
    metrics = {item.label: item.value for item in app_with_temp_artifacts.metric}
    assert set(metrics) >= {
        "Customers analysed",
        "Selected k",
        "Silhouette (clustering metric)",
        "Rows excluded while cleaning",
    }
    text = _texts(app_with_temp_artifacts)
    assert "How k was chosen" in text
    assert "How K-means works" in text
    assert "Assign a customer profile" in text
    assert "Assumptions and limitations" in text


def test_app_renders_the_charts_and_the_summary_table(app_with_temp_artifacts):
    app = app_with_temp_artifacts
    # footprint, sizes, elbow, silhouette
    assert _chart_count(app) == 4
    assert len(app.dataframe) == 2  # group summary + candidate metrics
    assert "not proof that a real category" in _texts(app)


def test_app_explains_that_there_is_no_accuracy_score(app_with_temp_artifacts):
    text = _texts(app_with_temp_artifacts)
    assert "no ground-truth segment labels" in text
    assert "arbitrary identifiers" in text
    assert "it does not predict future spending" in text


# ---------------------------------------------------------------------------
# The assignment form
# ---------------------------------------------------------------------------
def test_assignment_button_produces_a_result(app_with_temp_artifacts):
    app = app_with_temp_artifacts
    assert not app.exception

    assert len(app.get("form")) == 1, "the assignment form should be present"
    labels = [item.label for item in app.number_input]
    assert config.FEATURE_LABELS[config.INCOME_COLUMN] in labels
    assert config.FEATURE_LABELS[config.SPENDING_SCORE_COLUMN] in labels
    assert [item.label for item in app.button] == ["Assign to nearest cluster"]

    app.number_input[0].set_value(90.0).run()
    app.number_input[1].set_value(15).run()
    app.button[0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.success) == 1
    assert "Assigned to cluster" in app.success[0].value
    assert len(app.dataframe) == 3  # summary, candidates, distance-to-centroid table
    text = _texts(app)
    assert "standardised" in text
    assert "not a confidence score" in text
    assert "Cluster C" in text


def test_assignment_is_deterministic_and_matches_the_saved_model(
    app_with_temp_artifacts, trained_dirs
):
    from src.predict import assign_profile, load_artifacts

    app = app_with_temp_artifacts
    app.number_input[0].set_value(26.0).run()
    app.number_input[1].set_value(79).run()
    app.button[0].click().run()

    models_dir, _ = trained_dirs
    artifacts = load_artifacts(
        models_dir / config.MODEL_FILENAME, models_dir / config.METADATA_FILENAME
    )
    expected = assign_profile(artifacts, 26, 79)

    rendered = app.success[0].value
    assert f"C{expected.cluster_id}" in rendered
    assert expected.description in _texts(app)


def test_out_of_range_profile_triggers_a_warning(app_with_temp_artifacts):
    app = app_with_temp_artifacts
    # 300 k$ is inside the widget range but far above anything in the training data.
    app.number_input[0].set_value(300.0).run()
    app.number_input[1].set_value(50).run()
    app.button[0].click().run()

    assert not app.exception
    assert len(app.success) == 1
    joined = " ".join(item.value for item in app.warning)
    assert "outside the training range" in joined


def test_no_result_is_shown_before_the_button_is_pressed(app_with_temp_artifacts):
    app = app_with_temp_artifacts
    assert len(app.success) == 0
    assert len(app.error) == 0


# ---------------------------------------------------------------------------
# Missing artifacts
# ---------------------------------------------------------------------------
def test_missing_artifacts_show_setup_instructions_not_a_traceback(
    monkeypatch, tmp_path
):
    empty_models = tmp_path / "no-models"
    empty_outputs = tmp_path / "no-outputs"
    empty_models.mkdir()
    empty_outputs.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", empty_models)
    monkeypatch.setattr(config, "OUTPUTS_DIR", empty_outputs)

    app = AppTest.from_file(str(APP_FILE), default_timeout=DEFAULT_TIMEOUT).run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.error) == 1
    assert "not ready yet" in app.error[0].value
    text = _texts(app)
    assert "python -m src.train" in text
    assert "Mall_Customers.csv" in text
    assert "Kaggle" in text
    # The rest of the app must not render, because there is no model to describe.
    assert _chart_count(app) == 0


def test_incompatible_artifacts_show_a_friendly_message(monkeypatch, tmp_path, trained_dirs):
    models_dir, outputs_dir = trained_dirs
    metadata_file = models_dir / config.METADATA_FILENAME
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["artifact_schema_version"] = 42
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "OUTPUTS_DIR", outputs_dir)
    app = AppTest.from_file(str(APP_FILE), default_timeout=DEFAULT_TIMEOUT).run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.error) == 1
    assert "not ready yet" in app.error[0].value
    assert "schema version" in _texts(app)


# ---------------------------------------------------------------------------
# The app that ships with the real artifacts
# ---------------------------------------------------------------------------
def test_real_artifacts_app_renders_and_assigns(real_metadata):
    if not (config.MODELS_DIR / config.MODEL_FILENAME).is_file():
        pytest.skip("Real artifacts missing. Run: python -m src.train")
    if not (config.MODELS_DIR / config.FOOTPRINT_FILENAME).is_file():
        pytest.skip("Real footprint missing. Run: python -m src.train")

    app = AppTest.from_file(str(APP_FILE), default_timeout=DEFAULT_TIMEOUT).run()
    assert not app.exception, [str(e.value) for e in app.exception]

    app.number_input[0].set_value(
        float(round((real_metadata["features"]["training_ranges"][config.INCOME_COLUMN]["min"]
                    + real_metadata["features"]["training_ranges"][config.INCOME_COLUMN]["max"])
                   / 2, 1))
    ).run()
    app.number_input[1].set_value(50).run()
    app.button[0].click().run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.success) == 1
    assert "Assigned to cluster" in app.success[0].value


def _distance_table_value(app) -> "pd.DataFrame":
    """The distance-to-centroid table, located by its column signature.

    Selected by columns rather than by position, so adding or reordering the
    other tables in the app cannot make this test read the wrong one.
    """
    wanted = [
        "Cluster",
        "Description",
        "Centroid income (k$)",
        "Centroid spending score",
        "Distance (standardised units)",
    ]
    matches = [
        item.value for item in app.dataframe if list(item.value.columns) == wanted
    ]
    assert len(matches) == 1, f"expected exactly one distance table, found {len(matches)}"
    return matches[0]


def test_distance_table_shows_saved_centroids_not_distances(real_metadata):
    """Each column must hold the quantity its header names.

    Regression test: the distance table used to be built by looping over
    ``distances_standardized`` and writing that distance into the
    "Centroid income (k$)" cell, so the centroid column showed a duplicate of the
    distance. Centroids are now read from the fitted metadata by cluster id.
    """
    if not (config.MODELS_DIR / config.MODEL_FILENAME).is_file():
        pytest.skip("Real artifacts missing. Run: python -m src.train")

    app = AppTest.from_file(str(APP_FILE), default_timeout=DEFAULT_TIMEOUT).run()
    assert not app.exception, [str(e.value) for e in app.exception]

    app.number_input[0].set_value(80.0).run()
    app.number_input[1].set_value(50).run()
    app.button[0].click().run()
    assert not app.exception, [str(e.value) for e in app.exception]

    table = _distance_table_value(app)

    expected = {
        row[config.METADATA_CLUSTER_ID_KEY]: (
            row[config.METADATA_CENTROID_INCOME_KEY],
            row[config.METADATA_CENTROID_SCORE_KEY],
        )
        for row in real_metadata["clusters"]
    }
    assert len(table) == len(expected)

    for position, row in enumerate(table.to_dict("records")):
        cluster_id = int(row["Cluster"].removeprefix("C"))
        assert cluster_id == position, "rows should be ordered by cluster id"
        income, score = expected[cluster_id]

        # The saved centroid, matching the header, not the distance.
        assert row["Centroid income (k$)"] == pytest.approx(income, abs=1e-6)
        assert row["Centroid spending score"] == pytest.approx(score, abs=1e-6)
        assert row["Centroid income (k$)"] > 1.0, "a centroid income cannot be a distance"
        assert row["Description"] == real_metadata["segment_names"][f"C{cluster_id}"]

    # C0's real centroid: 55.3 k$ income and 49.5 spending score.
    c0 = table.to_dict("records")[0]
    assert c0["Centroid income (k$)"] == pytest.approx(55.3, abs=0.1)
    assert c0["Centroid spending score"] == pytest.approx(49.5, abs=0.1)

    # Distances come from the standardised-space calculation, and the shown
    # cluster is the minimum-distance one.
    distances = table.set_index("Cluster")["Distance (standardised units)"]
    assert len(distances) == len(expected)
    assert (distances > 0.0).all()
    nearest = distances.idxmin()
    assert f"cluster **{nearest}**" in app.success[0].value


def test_changing_the_profile_changes_distances_not_the_saved_centroids(real_metadata):
    """Centroids come from the fitted model, so no input can move them."""
    if not (config.MODELS_DIR / config.MODEL_FILENAME).is_file():
        pytest.skip("Real artifacts missing. Run: python -m src.train")

    def read_table(income: float, score: int) -> tuple[list, list]:
        app = AppTest.from_file(str(APP_FILE), default_timeout=DEFAULT_TIMEOUT).run()
        app.number_input[0].set_value(income).run()
        app.number_input[1].set_value(score).run()
        app.button[0].click().run()
        assert not app.exception, [str(e.value) for e in app.exception]
        table = _distance_table_value(app)
        centroids = table[["Centroid income (k$)", "Centroid spending score"]]
        return centroids.to_dict("records"), table[
            "Distance (standardised units)"
        ].tolist()

    first_centroids, first_distances = read_table(80.0, 50)
    second_centroids, second_distances = read_table(20.0, 20)

    assert first_centroids == second_centroids, "saved centroids must never change"
    assert first_distances != second_distances, "a different profile must change distances"


def test_no_rendered_markdown_contains_an_unescaped_dollar(app_with_temp_artifacts):
    """A bare ``$`` must never reach a markdown-rendering element.

    Streamlit renders markdown with KaTeX math switched on, so an unescaped
    ``$`` - as in the unit "k$" - can be swallowed as the opening delimiter of
    a math span and swallow the rest of the line. That is what produced the
    truncated "Spending0" sidebar label.

    This asserts the rendering-safety invariant rather than any wording: every
    ``$`` inside a rendered markdown string must be backslash-escaped.
    """
    offenders = []
    for element in app_with_temp_artifacts.markdown:
        value = element.value
        for index, char in enumerate(value):
            if char == "$" and (index == 0 or value[index - 1] != "\\"):
                offenders.append(value)
                break
    assert not offenders, f"unescaped '$' would render as math: {offenders}"


def test_both_feature_labels_appear_on_every_visible_surface(
    app_with_temp_artifacts,
):
    """The two features are labelled the same way wherever a reader sees them.

    Guards against the display label drifting from the config constant, which
    is how "Spending0" and a half-written "(1-100)" ended up on screen together.

    Widget labels are plain text, so they carry the label verbatim. Markdown
    carries the backslash-escaped form, which is what renders as the verbatim
    label once KaTeX is done with it.
    """
    app = app_with_temp_artifacts
    expected = set(config.FEATURE_LABELS.values())
    assert len(expected) == 2

    # Widget labels are plain text: exact match, no escaping.
    assert {item.label for item in app.number_input} == expected

    # Markdown surfaces render the escaped form.
    page = "\n".join(item.value for item in app.markdown)
    for label in expected:
        assert md(label) in page, f"label missing from the page: {label}"



# ---------------------------------------------------------------------------
# The hosted-deployment scenario: no dataset, no row-level export
# ---------------------------------------------------------------------------
def test_app_renders_without_the_dataset_or_the_row_level_export(
    monkeypatch, trained_dirs
):
    """A hosted copy gets no ``data/`` and no ``customer_segments.csv``.

    README section 10 promises the app still works from the committed artifacts
    alone, and that the local-only per-customer view disappears rather than
    breaking. Both are asserted here.
    """
    models_dir, outputs_dir = trained_dirs

    # Simulate .gitignore: drop the row-level export, keep the aggregate ones.
    (outputs_dir / config.SEGMENTS_FILENAME).unlink(missing_ok=True)

    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "OUTPUTS_DIR", outputs_dir)

    app = AppTest.from_file(str(APP_FILE), default_timeout=DEFAULT_TIMEOUT).run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.error) == 0

    # Every section still renders from models/ + the footprint alone. Headings
    # and expander labels are both counted, because the supporting explanation
    # is folded into expanders by design.
    sections = {item.value for item in app.header} | {
        item.label for item in app.expander
    }
    assert {
        "The groups that were found",
        "Assign a customer profile",
        "How k was chosen",
        "How K-means works",
        "Dataset details and cleaning policy",
        "Assumptions and limitations",
    } <= sections
    assert _chart_count(app) >= 3
    assert len(app.dataframe) >= 1

    # No per-customer control is offered, so no individual customer can be
    # displayed. (The privacy note may still mention "row-level" in order to
    # explain why the view is absent.)
    assert app.checkbox == []

    # Assignment still works without the CSV.
    app.number_input[0].set_value(60.0).run()
    app.number_input[1].set_value(50).run()
    app.button[0].click().run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.success) == 1
    assert "Assigned to cluster" in app.success[0].value


def test_app_explains_the_limitation_and_disclaims_rfm(app_with_temp_artifacts):
    """The app must state the dataset's real limitation on screen.

    Guards against the failure mode where a marketing sentence quietly implies
    transaction history the dataset does not contain. The app is expected to
    *name* recency/RFM in order to rule them out, so this checks for the
    disclaimers rather than for the mere absence of the words.
    """
    text = _texts(app_with_temp_artifacts).lower()

    # The limitation is stated plainly.
    assert "one row per membership customer" in text
    assert "no dated transactions" in text
    assert "not a measured purchase history" in text

    # RFM and purchase history are explicitly ruled out rather than implied.
    assert "does not compute rfm" in text or "not compute rfm" in text
    assert "does not predict" in text

    # No accuracy claim of any kind.
    for forbidden in ("accuracy of", "we achieved", "model accuracy"):
        assert forbidden not in text, f"app makes an unsupported claim: {forbidden}"
