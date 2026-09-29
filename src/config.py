"""Project-wide settings.

Everything the rest of the project needs to agree on lives here: file paths,
the two clustering features, their units, and the K-means hyper-parameters.

All paths are derived from this file's location, so the project folder can be
copied or cloned anywhere and the code still works.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "Mall_Customers.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Trusted artifacts. These are written by ``python -m src.train`` and are the
# only files the Streamlit app is allowed to deserialise.
MODEL_FILENAME = "kmeans_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"
FOOTPRINT_FILENAME = "segment_footprint.json"

# Generated result files.
SEGMENTS_FILENAME = "customer_segments.csv"
SUMMARY_FILENAME = "cluster_summary.csv"
K_METRICS_FILENAME = "k_selection_metrics.csv"

# Bumped whenever the artifact layout changes; ``src.predict`` refuses to load
# artifacts produced under a different version.
ARTIFACT_SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# Dataset attribution (verified from the Kaggle dataset page and its public
# metadata endpoint on the date recorded in the README)
# --------------------------------------------------------------------------
DATASET_TITLE = "Mall Customer Segmentation Data"
DATASET_URL = (
    "https://www.kaggle.com/datasets/vjchoudhary7/"
    "customer-segmentation-tutorial-in-python"
)
DATASET_AUTHOR = "Vijay Choudhary (Kaggle user: vjchoudhary7)"
DATASET_LICENSE = (
    "Other (specified in description) - the Kaggle dataset page does not state a "
    "standard open licence, so the raw CSV is treated as not redistributable."
)
DATASET_ACKNOWLEDGEMENT = (
    "The dataset page credits the file to the Udemy 'Machine Learning A-Z' course "
    "and describes it as created for learning customer segmentation concepts "
    "(market basket analysis). Spending Score is described on the source page as a "
    "value the mall owner assigns to a customer from their own parameters, "
    "customer behaviour and purchasing data."
)
DATASET_LIMITATION = (
    "This file is a customer-level profile table. It has one row per membership "
    "customer and no dated transactions, so it cannot tell us what a customer "
    "bought, how often, or when they last bought. Spending Score is a single "
    "assigned summary number, not a measured purchase history."
)

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
ID_COLUMN = "CustomerID"
GENDER_COLUMN = "Gender"
AGE_COLUMN = "Age"
INCOME_COLUMN = "Annual Income (k$)"
SPENDING_SCORE_COLUMN = "Spending Score (1-100)"

#: The only two columns the clustering model is allowed to see, in this order.
FEATURE_COLUMNS: tuple[str, str] = (INCOME_COLUMN, SPENDING_SCORE_COLUMN)

FEATURE_UNITS: dict[str, str] = {
    INCOME_COLUMN: "k$ (thousands, as labelled by the dataset source)",
    SPENDING_SCORE_COLUMN: "score from 1 to 100 (assigned by the dataset source)",
}

#: Human-readable labels for on-screen display. These are deliberately *not*
#: the dataset column names: they read as prose ("Annual income (k$)") instead
#: of a column header, and they keep the two apart so a raw column name never
#: leaks into a widget label or a sentence.
FEATURE_LABELS: dict[str, str] = {
    INCOME_COLUMN: "Annual income (k$)",
    SPENDING_SCORE_COLUMN: "Spending score (1\u2013100)",
}

#: Needed to trace a local result back to the source CSV.
REQUIRED_COLUMNS: tuple[str, ...] = (ID_COLUMN, *FEATURE_COLUMNS)

#: Present in this dataset but deliberately kept out of the model.
OPTIONAL_DEMOGRAPHIC_COLUMNS: tuple[str, ...] = (GENDER_COLUMN, AGE_COLUMN)

EXCLUDED_FROM_MODEL: tuple[str, ...] = (ID_COLUMN, *OPTIONAL_DEMOGRAPHIC_COLUMNS)

#: Column added by the cleaning step so a local result can be traced to the CSV.
SOURCE_ROW_COLUMN = "source_row"

# --------------------------------------------------------------------------
# Metadata column names for the per-cluster summary
# --------------------------------------------------------------------------
#: One place to name these, so the training script, the saved metadata, the app
#: and the inference code can never drift apart.
METADATA_CLUSTER_ID_KEY = "cluster_id"
METADATA_CUSTOMERS_KEY = "customers"
METADATA_DESCRIPTION_KEY = "description"
METADATA_COLOR_KEY = "color"
METADATA_CENTROID_INCOME_KEY = "centroid_income_k"
METADATA_CENTROID_SCORE_KEY = "centroid_spending_score"
METADATA_MEAN_INCOME_KEY = "mean_income_k"
METADATA_MEAN_SCORE_KEY = "mean_spending_score"

# --------------------------------------------------------------------------
# Validation limits
# --------------------------------------------------------------------------
INCOME_MIN = 0.0
SPENDING_SCORE_MIN = 1.0
SPENDING_SCORE_MAX = 100.0

#: A clustering run is refused below these numbers so the metrics stay meaningful.
MIN_ROWS_FOR_CLUSTERING = 20
MIN_DISTINCT_FEATURE_PAIRS = 5

# --------------------------------------------------------------------------
# K-means settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42
N_INIT = 20
K_MIN = 1
K_MAX = 10
SILHOUETTE_K_MIN = 2

#: A centroid is called "higher"/"lower" when it sits this far from the
#: dataset mean, as a fraction of that mean.
RELATIVE_TOLERANCE = 0.10

#: Stable colours. Cluster ``i`` always uses ``CLUSTER_COLORS[i]`` in the app,
#: in the saved figures and in the CSV exports.
CLUSTER_COLORS: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

#: Resolution of the aggregate footprint grid saved for deployment. Binned
#: counts only - no individual customer row is recoverable from it.
FOOTPRINT_INCOME_BINS = 24
FOOTPRINT_SPENDING_BINS = 20

PROJECT_TITLE = "PRODIGY_ML_02 - Customer Segmentation Using K-means"
APP_TITLE = "Customer Segmentation Explorer"

DATASET_SETUP_INSTRUCTIONS = f"""The dataset file was not found.

1. Open {DATASET_URL}
2. Sign in to Kaggle and accept the dataset terms.
3. Download the archive and extract `Mall_Customers.csv` into:

   {DATA_FILE}

4. Create the model artifacts:

   python -m src.train

Only download the file from the official Kaggle page. Do not substitute a
mirror or an unofficial copy, because the terms of those copies are unknown."""
