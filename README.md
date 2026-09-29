# PRODIGY_ML_02 — Customer Segmentation Using K-means

**Prodigy InfoTech — Machine Learning Task 02**

Retail customer records are grouped by income and spending behaviour using
K-means, the groups are described from the numbers the model actually measured,
and a Streamlit app lets you assign a profile to a group interactively.

---

## 1. Assignment objective

> Group retail customers based on their spending behaviour using K-means
> clustering.

Concretely: build a reproducible pipeline that standardises the two relevant
features, chooses a defensible number of clusters from the data, fits one
`StandardScaler` + `KMeans` pipeline, and serves it in a Streamlit app — without
over-claiming what the dataset can support.

---

## 2. Dataset source and manual download

**Mall Customer Segmentation Data** by Vijay Choudhary (Kaggle user
`vjchoudhary7`)
<https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python>

Kaggle authentication is required, so the CSV is **not** shipped with this
repository. Download it manually:

1. Sign in to Kaggle and open the dataset page.
2. Accept the dataset terms.
3. Click **Download** and unzip the archive.
4. Copy `Mall_Customers.csv` into:

   ```
   PRODIGY_ML_02\data\Mall_Customers.csv
   ```

Use only the official Kaggle file. Do not substitute a mirror or an unofficial
copy — the terms of those copies are unknown.

### Attribution and third-party notices (kept as the source states them)

* The dataset page describes the file as "created only for the learning purpose
  of the customer segmentation concepts, also known as market basket analysis".
* The page's **Acknowledgements** read, verbatim: "From Udemy's Machine
  Learning A-Z course." and "I am new to Data science field and want to share my
  knowledge to others", followed by a link to that course's copy of
  `Mall_Customers.csv` in the `SteffiPeTaffy/machineLearningAZ` repository.
* The page's own definition of the key field: Spending Score "is something you
  assign to the customer based on your defined parameters like customer behavior
  and purchasing data."

### Licensing: what is verifiable, and what is not

Verified from Kaggle's own public metadata endpoint for this dataset
(`https://www.kaggle.com/api/v1/datasets/view/vjchoudhary7/customer-segmentation-tutorial-in-python`):

| Field | Value |
| --- | --- |
| Dataset id | `42674` |
| `ref` | `vjchoudhary7/customer-segmentation-tutorial-in-python` |
| Title / subtitle | `Mall Customer Segmentation Data` / `Market Basket Analysis` |
| Owner | `Vijay Choudhary` (`vjchoudhary7`) |
| `licenseName` | **`Other (specified in description)`** |
| Version | `1`, version notes `Initial release` |
| `lastUpdated` | `2018-08-11T07:23:02.83Z` |
| Public | `isPrivate: false` |

**The licence field points at a specification that is not there.** "Other
(specified in description)" means the terms should be written in the dataset
description. Searching that description returns **zero** occurrences of
`licen`, `creative commons`, `cc`, `attribution`, `copyright`, `redistribut`,
`commercial`, `free to`, `public domain`, `mit`, `apache`, `gpl` or
`permission`.

So the honest position is:

| | |
| --- | --- |
| **Explicit permission to redistribute** | **Not found.** No licence text exists to grant it. |
| **Explicit prohibition on redistribution** | **Not found either.** Nothing forbids it. |
| **Actual state** | The terms are **unclear**. The pointer to them is broken. |

This project does **not** claim permission, and it does **not** declare the
dataset off-limits for everyone. It takes the cautious route of not
redistributing the file, because an absent grant is not a grant. Re-check the
page before publishing anything. See
[§10 Publishing](#10-what-may-be-published) for what is and is not in the
intended bundle.

---

## 3. The actual features and their units

| Column | In the file | Role in this project | Unit |
| --- | --- | --- | --- |
| `CustomerID` | yes | Identifier, kept for traceability. **Not a model input.** | — |
| `Gender` | yes | Present, **not a model input.** | — |
| `Age` | yes | Present, **not a model input.** | — |
| `Annual Income (k$)` | yes | **Clustering feature** | k$ = thousands, exactly as the source labels it |
| `Spending Score (1-100)` | yes | **Clustering feature** | whole-number score, 1 to 100 |

The two predictors are the only columns the model ever sees. `CustomerID` is an
identifier rather than behaviour, and leaving gender and age out keeps the
groups defined purely by income and spending score. The app and the README never
describe the groups in terms of age, gender or loyalty, because the model has no
information about those.

Units are kept exactly as the dataset states them. The file is not labelled with
a currency for any particular country, so this project does **not** call the
income INR and does **not** describe the customers as current Indian customers.

### Dataset limitation: there is no transaction-level history

**This is a customer-level profile table, not a transaction log.** There is one
row per membership customer and no dates, no line items and no amounts, so the
dataset cannot answer *what* someone bought, *how often*, or *when they last
bought*.

Because of that:

* Spending Score is treated as **one assigned summary number**, which is exactly
  what the source page says it is.
* The project does **not** claim to analyse transaction history, purchase
  frequency or recency.
* **No RFM features are computed**, because the data to compute them does not
  exist here. Inventing them would mean inventing customer behaviour.

---

## 4. Cleaning policy

Implemented in `src/data.py`, reported in the training output and saved into
`models/model_metadata.json`.

* **Required columns are checked** before anything else. A missing
  `CustomerID`, `Annual Income (k$)` or `Spending Score (1-100)` stops the run
  with the list of what it found.
* **Complete cases only.** A row is kept only when both predictors are numeric
  and finite, income is ≥ 0 k$, and the spending score is a whole number from 1
  to 100. **Nothing is imputed** — an invented income or score would be an
  invented customer. Every excluded row is counted by reason.
* **Enough data to cluster.** Fewer than 20 valid rows, or fewer than 5 distinct
  income/score pairs, stops the run instead of producing meaningless metrics.
* **Customers who look alike are all kept.** Two customers with the same income
  *and* the same spending score are two different people, so neither is removed.
  The count of such coincidences is reported.
* **Only records identical in *every* column** are treated as duplicated records;
  those are dropped keeping the first occurrence, and the count is reported.
* **Traceability.** Every kept row keeps a 1-based `source_row` number for the
  CSV and the dataset's own `CustomerID`.

### What the cleaning actually found in this file

| Measure | Value |
| --- | --- |
| Rows in the file | 200 |
| Rows used | 200 |
| Rows excluded | 0 (no invalid rows) |
| Exact duplicate records removed | 0 |
| Customers sharing income **and** spending score (all kept) | 4 |
| File SHA-256 | `416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77` |

---

## 5. Scaling and how k was chosen

### Scaling

K-means minimises the sum of squared **distances**, and squared distance weights
a column by its variance. `StandardScaler` gives both features comparable
variance so their numerical scales do not disproportionately influence
distances: it subtracts each feature's mean and divides by its standard
deviation.

**Whether an unscaled feature dominates depends on its actual numerical spread,
not on what its unit is called.** A column labelled "k$" is not special, and
renaming a column changes nothing. What matters is the spread of the numbers in
it.

For this dataset the two spreads happen to be similar — standard deviations of
about 26.3 for income and 25.8 for spending score — so scaling is a safeguard
here rather than a correction, and it does not dramatically change the answer.
It is still the right default, and it changes the answer substantially on data
whose columns are recorded on very different scales.

Both sides of that statement are tested in `tests/test_clustering.py`:

* `test_scaling_matters_because_units_differ` builds columns with genuinely
  different spread (income varying by thousands, spending score by tens) and
  shows the unscaled fit chasing the wrong column.
* `test_scaling_is_harmless_when_spreads_are_similar` builds columns with
  comparable spread and shows the unscaled and standardised fits agreeing.
* `test_standard_scaler_subtracts_the_mean_and_divides_by_the_std` checks the
  transform numerically instead of trusting a description of it.

### Metrics

* **Inertia** — within-cluster sum of squares. Recorded for k = 1 to 10. It
  always falls as k rises, so it cannot pick k on its own. The elbow curve is
  shown as **supporting evidence only**.
* **Silhouette score** — for each customer, compares the distance to its own
  group with the distance to the nearest other group. Higher means more
  separated, more internally tight groups. It is a **clustering metric, not a
  prediction accuracy**, and it is only defined for k ≥ 2.

Both are measured in the **same standardised matrix the K-means model learns
from**. Candidates that cannot be scored (k above the sample size, k above the
number of distinct feature pairs, or a fit that collapsed into fewer groups)
are marked with a note and skipped instead of crashing.

**Selection rule: the highest measured silhouette score among valid candidates
wins, and an exact tie goes to the smaller k.** k is *not* hardcoded to 5.

### Measured results on this dataset

| k | Inertia | Silhouette | Note |
| ---: | ---: | ---: | --- |
| 1 | 400.00 | n/a | silhouette needs at least 2 clusters |
| 2 | 269.69 | 0.3213 | |
| 3 | 157.70 | 0.4666 | |
| 4 | 108.92 | 0.4939 | |
| **5** | **65.57** | **0.5547** | **selected** |
| 6 | 55.06 | 0.5399 | |
| 7 | 44.86 | 0.5281 | |
| 8 | 37.15 | 0.4567 | |
| 9 | 32.39 | 0.4571 | |
| 10 | 29.69 | 0.4362 | |

**Selected k = 5** (silhouette **0.5547**, inertia **65.57**).
Settings: `random_state=42`, `n_init=20`.

### The five groups (measured centroids, original units)

| Cluster | Customers | Centroid income (k$) | Centroid score | Description (relative to this dataset) |
| --- | ---: | ---: | ---: | --- |
| C0 | 81 | 55.30 | 49.52 | mid-range income, mid-range spending score |
| C1 | 39 | 86.54 | 82.13 | higher income, higher spending score |
| C2 | 22 | 25.73 | 79.36 | lower income, higher spending score |
| C3 | 35 | 88.20 | 17.11 | higher income, lower spending score |
| C4 | 23 | 26.30 | 20.91 | lower income, lower spending score |

Cluster counts sum to 200, the cleaned row count. Descriptions are generated
mechanically: a centroid is called *higher* or *lower* than a feature when it sits
more than 10% away from this dataset's mean for that feature. The wording is
deliberately neutral and the mapping `C0…C4 → description` is persisted in
`models/model_metadata.json` so a cluster id always means the same thing
everywhere.

### What these numbers are, and what they are not

* **k = 5 is a practical choice inside the tested range 1–10.** It is not a
  discovered truth, and it was not chosen because a tutorial used the same
  number.
* **The silhouette score is not an accuracy.** It measures separation and
  cohesion, not how right the grouping is.
* **There are no ground-truth segment labels**, so accuracy, precision, recall,
  F1 and R² do not exist for this task. None of them are reported anywhere in
  this project.
* **All metrics are descriptive and were measured on the dataset the model was
  fitted on.** This is *not* held-out supervised validation, and it does not
  promise the groups will hold for new customers.
* **Cluster numbers are arbitrary identifiers**, not a ranking of value.
  Renumbering them would describe exactly the same grouping.
* **A 2D cluster boundary is an algorithmic grouping**, not proof that a real
  "type" of customer exists.

---

## 6. Local setup and run commands (Windows PowerShell)

Run everything from the project folder
(`...\Documents\Projects\PRODIGY_ML_02`).

```powershell
# 1. Create an isolated environment (Python 3.12 was used here)
py -3.12 -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1
#    If PowerShell blocks the activation script, allow it for this user once:
#    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 3. Install runtime dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. (Optional, for tests and the notebook)
python -m pip install -r requirements-dev.txt

# 5. Put Mall_Customers.csv in .\data\   (see section 2)

# 6. Train and write the artifacts
python -m src.train

# 7. Assign a sample profile from the command line
python -m src.predict --income 82 --score 91

# 8. Run the tests
python -m pytest

# 9. Start the app
python -m streamlit run app.py
```

Without activating, replace `python` with `.\.venv\Scripts\python.exe`.

---

## 7. File structure and regenerating the artifacts

```
PRODIGY_ML_02/
├── app.py                                    Streamlit app
├── src/
│   ├── __init__.py
│   ├── config.py                             paths, feature names, units, constants
│   ├── data.py                               loading, validation, cleaning
│   ├── clustering.py                         K-means sweep, k selection, summaries
│   ├── train.py                              training entry point (python -m src.train)
│   ├── predict.py                            artifacts + single-profile assignment
│   └── visualization.py                      every matplotlib figure
├── tests/
│   ├── conftest.py                           fixtures (synthetic data, real artifacts)
│   ├── test_data.py                          columns, numeric validation, cleaning
│   ├── test_clustering.py                    scaling, k sweep, selection, centroids
│   ├── test_predict.py                       load/assign, validation, consistency
│   └── test_app.py                           Streamlit AppTest
├── notebooks/
│   └── customer_segmentation_walkthrough.ipynb
├── data/                                     Mall_Customers.csv goes here (git-ignored)
├── models/                                   trusted artifacts (published)
│   ├── kmeans_pipeline.joblib                fitted scaler + KMeans
│   ├── model_metadata.json                   everything recorded at training time
│   └── segment_footprint.json                aggregate binned counts for the app
├── outputs/
│   ├── customer_segments.csv                 row level, local only (git-ignored)
│   ├── cluster_summary.csv                   aggregate, committed
│   ├── k_selection_metrics.csv               aggregate, committed
│   └── figures/                              PNG charts, local only (git-ignored)
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .gitignore
├── .streamlit/config.toml
└── README.md
```

`src/config.py` and `src/predict.py` were split out of the suggested layout so
that paths, feature names and units are declared once, and so the app, the tests
and the CLI share one inference path. Everything else follows the suggested
structure.

### Regenerating the artifacts

```powershell
python -m src.train                      # full run, including PNG figures
python -m src.train --k-max 8            # test a narrower search range
python -m src.train --no-figures         # skip the PNG step
python -m src.train --data path\to\Mall_Customers.csv
```

Each run rewrites `models/` and `outputs/`. The dataset SHA-256, row counts,
exclusions, training ranges, timestamp and dependency versions are written into
`models/model_metadata.json`, so a result can always be tied to an exact input
file.

---

## 8. The Streamlit app

`python -m streamlit run app.py` opens **Customer Segmentation Explorer**.

The main workflow is kept at the top, and the supporting explanation is folded
into named expanders so the useful part is reachable without scrolling past
detail.

**Sidebar** — three expanders: *At a glance* (customers, selected k, silhouette,
the two feature labels, the excluded columns), *Dataset* (title, author, licence
as shown on the source page, rows used, Kaggle link, filename, hash), *Run it
locally*.

**Main page, in order:**

1. Title, one-line task summary, and the four key metrics (customers analysed,
   selected k, silhouette, rows excluded).
2. A short standing notice that this dataset holds **customer profiles and an
   assigned spending score, not transaction histories**.
3. **The groups that were found** — the aggregate footprint chart, the group
   summary table, cluster sizes, and (locally only) the per-customer scatter
   behind an unchecked checkbox.
4. **Assign a customer profile** — the form, with the "this does not predict
   future spending" statement directly above it.
5. Expanders, closed by default: *How k was chosen* (elbow, silhouette, candidate
   table, and the silhouette-is-not-accuracy explanation), *How K-means works*,
   *Dataset details and cleaning policy*, *Technical details*.
6. **Assumptions and limitations** — one consolidated section, including the
   standing notice about what the dataset is not.
7. The source acknowledgement.

**The app never retrains.** It reads the artifacts from a fixed path, caches them
with `@st.cache_resource` / `@st.cache_data`, and calls the same
`src.predict` functions the tests and the CLI use. If the artifacts are missing
or were written by a different version of the project, the app shows local setup
instructions instead of a traceback.

### How labels and units are rendered

The two features are shown with display labels, not raw column names:
**Annual income (k$)** and **Spending score (1–100)**. The underlying dataset
columns keep their original spellings (`Annual Income (k$)` and
`Spending Score (1-100)`) and are never rewritten.

Streamlit renders markdown with KaTeX math enabled, so a bare `$` in prose — as
in the unit "k$" — can be swallowed as the opening delimiter of a math span and
garble the rest of the line. That is what truncated the sidebar feature label to
`Spending0`. Every `$` passed to a markdown element is therefore backslash
escaped by the `md()` helper in `app.py`, which renders a literal `$`.
`test_no_rendered_markdown_contains_an_unescaped_dollar` fails if any unescaped
`$` ever reaches a markdown element again.

Widget labels, tooltips and dataframe headers are plain text and are deliberately
*not* escaped, because escaping those would show a stray backslash.

### About the assignment form

The form takes an annual income in k$ and a spending score from 1 to 100, then
assigns that profile to the nearest learned cluster and shows the cluster id, the
description and the centroid.

* **Validation lives in `src/predict.py`**, not only in the widgets, so the app,
  the CLI and the notebook accept and reject exactly the same values.
* The **fitted scaler and model** are reused. A new scaler is never fitted.
* This **assigns an existing spending score to a group. It does not predict
  future spending**, and the app says so above the form.
* Inputs **outside the observed training range** still get an assignment (the
  nearest centre is the model's only answer) but produce a clear warning.
* **No confidence percentages and no probabilities.** The only number shown
  besides the distance is nothing: the distance to each centroid is labelled as
  *standardised feature space* and explained as a geometric distance, explicitly
  not a confidence score.
* There is no file upload, no login, no customer tracking, no database and no
  paid API.

---

## 9. Tests and verification

```powershell
python -m pytest              # whole suite
python -m pytest -v           # with test names
```

`pytest.ini` sets `testpaths = tests`. Synthetic data is used **only** inside
automated tests and is always labelled as synthetic; no test result is presented
as a real-data result.

The suite covers:

| Area | Where |
| --- | --- |
| Required columns, missing file, numeric validation | `test_data.py` |
| Missing/invalid rows counted by reason, source-row order preserved | `test_data.py` |
| Duplicate feature pairs kept, exact duplicates removed | `test_data.py` |
| Too-few-rows and constant-feature refusal | `test_data.py` |
| `CustomerID`/`Gender`/`Age` excluded from the predictors | `test_clustering.py` |
| Feature order fixed; fitted scaler is standardised | `test_clustering.py` |
| Scaling matters when the columns' **spread** differs wildly | `test_clustering.py` |
| Scaling is harmless when the spread is similar (the honest counterweight) | `test_clustering.py` |
| The scaler is exactly mean-subtraction / std-division, checked numerically | `test_clustering.py` |
| Invalid k, k above the sample size, degenerate data | `test_clustering.py` |
| k selection, including exact ties going to the smaller k | `test_clustering.py` |
| Centroids in original units, counts summing to the row count | `test_clustering.py` |
| Artifact load failures, schema/feature mismatch, corrupt file | `test_predict.py` |
| Save/load prediction equivalence; no refit on prediction | `test_predict.py` |
| Cluster id, description and centroid consistent across outputs | `test_predict.py` |
| Out-of-range warning behaviour | `test_predict.py` |
| Single-profile input validation (9 rejected cases: negative income, score 0/101/fractional/NaN/Inf, non-numeric string, `None`, and `bool`) | `test_predict.py` |
| Real-artifact checks, skipped if training has not been run | `test_predict.py` |
| AppTest: main sections, charts, tables | `test_app.py` |
| AppTest: the **Assign to nearest cluster** button and its result | `test_app.py` |
| AppTest: missing artifacts and incompatible artifacts | `test_app.py` |
| AppTest: the app that ships with the real artifacts | `test_app.py` |
| AppTest: renders with **no `data/` and no row-level CSV** (hosted scenario) | `test_app.py` |
| AppTest: no unescaped `$` reaches any markdown element | `test_app.py` |
| AppTest: both feature labels used on every visible surface | `test_app.py` |
| AppTest: states the dataset limitation and disclaims RFM / accuracy | `test_app.py` |

`tests/test_app.py` uses Streamlit's own `AppTest` runner against the real
`app.py`, so it exercises the actual button, not a mock.

The hosted-deployment row is the important one: it deletes
`customer_segments.csv`, runs the app, and asserts that every section still
renders, that the per-customer checkbox is gone, and that assignment still works
from `models/` alone. That is what stops the "works on my machine" version of
this project from breaking after the dataset is left out of the repository.

Two presentation tests guard against the rendering bugs that were actually
observed:

* `test_no_rendered_markdown_contains_an_unescaped_dollar` — asserts the
  invariant that every `$` reaching a markdown element is backslash-escaped, so
  KaTeX cannot swallow the `k$` unit and truncate a line.
* `test_both_feature_labels_appear_on_every_visible_surface` — asserts both
  display labels are used on the widgets and in the markdown, so the two cannot
  drift apart again.

Neither test checks wording; both check structure and behaviour.

### The walkthrough notebook

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace `
  notebooks\customer_segmentation_walkthrough.ipynb
```

The notebook covers dataset exploration and limitations, feature selection and
scaling, choosing k, fitting and visualising the clusters, interpreting the
centroids, and assigning a new profile. It calls the project functions rather
than re-implementing them, so its numbers match the app's. To open it:

```powershell
python -m jupyter lab notebooks\customer_segmentation_walkthrough.ipynb
```

### What was verified

| Check | Result |
| --- | --- |
| `python -m pytest` | **92 passed** |
| `python -m src.train` on the real CSV | 200/200 rows used, selected k = 5 |
| Model artifacts after the presentation work | **unchanged** — k = 5, silhouette 0.5547, inertia 65.57, all five centroids and counts identical, artifacts not regenerated |
| `--income 60 --score 50` | still **C0**, "mid-range income, mid-range spending score", centroid 55.3 k$ / 49.5, distance 0.1805 |
| Notebook executed end to end (`nbconvert --execute`) | 16 code cells, **0 errors**, 7 figures rendered |
| `python -m pip_audit` | **No known vulnerabilities** |
| Streamlit server starts and serves | `/_stcore/health` → `200 ok`, `/` → `200` |
| App with no `data/` and no row-level CSV | 0 exceptions, 0 errors, all sections present, row-level checkbox correctly absent |
| `segment_footprint.json` structure | walked in full: 7 leaves, bin edges + aggregate counts only, no IDs, sums to 200 |
| Every generated PNG | non-blank, readable dimensions |
| `src.predict` CLI | correct cluster, distance and validation exit codes |

**Visual verification is still manual.** No browser automation tool is available
in this environment, so the app has **not** been looked at in a real browser.
AppTest confirms which widgets and strings are produced, and HTTP 200 confirms
the server serves, but neither one renders the page. The desktop and
narrow-screen layout, the sidebar labels, and the scaling paragraph in
*How K-means works* all still need a human eye — in particular, that
`Annual income (k$)` and `Spending score (1–100)` appear intact and that no
`$` is rendered as math. The `$`-escaping invariant is covered automatically;
its *appearance* is not.

---

## 10. What may be published

The licence is **"Other (specified in description)"** and the description
contains no licence terms at all (see
[§2 Licensing](#licensing-what-is-verifiable-and-what-is-not)). There is no
explicit grant of redistribution rights and no explicit prohibition either, so
this project does not assume a right to redistribute the data. The split:

**Not published, stays local**

* `data/Mall_Customers.csv` (the dataset itself)
* `outputs/customer_segments.csv` (one row per customer — it *is* the dataset)
* `outputs/figures/cluster_scatter_local_only.png` (a scatter of every customer's
  coordinates is dataset redistribution regardless of the file format)
* `outputs/figures/` in general

These are all excluded by `.gitignore`, and
`test_app_renders_without_the_dataset_or_the_row_level_export` fails if the app
starts depending on them.

**Published (aggregate only)**

* `models/kmeans_pipeline.joblib` — the fitted scaler and centroids
* `models/model_metadata.json` — features, k, metrics, centroids, hashes
* `models/segment_footprint.json` — **binned counts only**, schema below
* `outputs/cluster_summary.csv`, `outputs/k_selection_metrics.csv` — aggregates

The aggregate cluster chart is not shipped as a PNG. The app draws it at runtime
from `models/segment_footprint.json`, so the hosted app shows the aggregate view
with no need for either the CSV or the figures folder.

This is why `src/visualization.py` has two scatter views.
`plot_cluster_scatter` draws one dot per customer and is the local teaching view.
`plot_cluster_footprint` draws only binned counts plus the centroids, and it is
the default in the app. The row-level view sits behind an unchecked checkbox, is
clearly labelled local-only, and disappears entirely when the row-level CSV is
absent — which is the situation in a hosted deployment.

### What `models/segment_footprint.json` actually contains

Seven top-level keys, no nested objects:

| Key | Type | Value |
| --- | --- | --- |
| `selected_k` | int | `5` |
| `n_customers` | int | `200` |
| `source` | str | `Mall_Customers.csv (sha256 416a4f62a8f3...)` |
| `income_bin_edges` | list[25] float | 15.0 → 137.0, evenly spaced (24 bins, width 5.0833) |
| `spending_score_bin_edges` | list[21] float | 1.0 → 99.0, evenly spaced (20 bins, width 4.9) |
| `counts` | list[24][20] int | aggregate count per bin |
| `note` | str | states that it is binned counts only |

**Does it contain bin edges and aggregate counts?** Yes — that is all it is.

**Does it contain individual customers?** No. A full structural walk of the file
yields exactly the seven leaves above. There is no `CustomerID`, no `source_row`,
no `Gender`, no `Age`, no per-customer coordinates and no embedded row-level
records. The only reference to the dataset is its filename and a truncated hash
in `source`.

**Do the counts reconcile?** Yes, exactly:

```
sum(counts)        = 200
n_customers        = 200
metadata rows_kept = 200  (of 200 in the file)
metadata training_rows = 200
```

The bin edges are evenly spaced grid lines that happen to start and end at the
observed minimum and maximum, so the file does disclose that the data spans
15–137 k$ and 1–99. That is an aggregate fact, not a record.

### What aggregation does *not* do

Binning the counts is a precaution and a smaller disclosure. It is **not** a
licence decision and **not** a guarantee of anonymity, for two concrete reasons
found by inspecting this file:

* **78 of the 480 bins hold exactly one record** (16.2%), and 98 hold one or two.
  A singleton bin contains one dataset record within an income/spending interval.
  This alone does not establish that the record identifies a real person.
  Aggregation does not guarantee anonymity or resolve redistribution permissions.
* The grid is built from the same cleaned rows, so its range and shape are
  derived from the dataset.

So the honest statement is: *the footprint exposes no customer identifiers and no
row-level records, and aggregation does not guarantee anonymity or resolve
redistribution permissions.* The dataset's redistribution terms remain unclear.
Raw data and row-level exports are excluded from the publication bundle as a
precaution, as is the per-customer chart. That is a precaution under uncertainty,
not a claim that publication is prohibited or that these derived artifacts are
licensed for redistribution.

**Before publishing anything, re-check the current terms on the Kaggle page.**
If the terms are clarified later, revisit this section.

### Deployment-artifact strategy

A hosted copy of this app must not depend on the ignored CSV and must not train
at startup. It does not:

* `data/` is git-ignored and absent in a hosted deployment.
* The app reads `models/kmeans_pipeline.joblib` and `models/model_metadata.json`,
  which **are** committed.
* Every chart is rebuilt from those artifacts plus
  `models/segment_footprint.json`, so the cluster view still renders without the
  CSV.
* The app never calls `src.train`; training is a separate, explicit command.

### Deploying to Streamlit Community Cloud

This section is the verified setup for a free Community Cloud deployment. **The
hosted app has not been created yet** - see the status note at the end.

Sign in at [share.streamlit.io](https://share.streamlit.io), choose **Create
app**, then **Yup, I have an app**, and enter:

| Field | Value |
| --- | --- |
| Repository | `an-codes1/PRODIGY_ML_02` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL (subdomain) | `amber-prodigy-ml-02` |
| Python version | `3.12` (also the Community Cloud default) |
| Secrets | none needed |

Python 3.12 is the version the committed artifacts were fitted with, and it is
the version the `.github/workflows/ci.yml` workflow tests on, so the deployed
runtime matches what the tests already cover. There are no secrets to enter: the
app reads only committed files.

The resulting URL will be:

```
https://amber-prodigy-ml-02.streamlit.app
```

**Status: not yet deployed.** Creating the app needs a signed-in browser session
and a GitHub authorisation, neither of which can be done from a terminal. The
repository is published and CI-verified, so the remaining work is the few clicks
above. Once the URL is live, this section should be updated with the confirmed
link and the checks in §9 extended with what was verified on the hosted app.

---

## 11. Security notes

* The **only** file deserialised is the project-generated
  `models/kmeans_pipeline.joblib`, read from a fixed path. No user upload is ever
  unpickled.
* `src/predict.py` catches a failed `joblib.load` and reports a friendly message
  with the re-run command instead of letting a traceback escape.
* No `eval`, no `exec`, no user-controlled shell commands, and no
  `unsafe_allow_html`. Widget values are parsed as numbers and validated.
* `.streamlit/config.toml` keeps `enableXsrfProtection = true`,
  `enableCORS = true` and `xsrfCookieSameSite = "lax"`, and sets
  `enableStaticServing = false`. All of these are supported options in Streamlit
  1.64, and they are Streamlit's defaults written out explicitly so nobody
  disables them by accident.
* `.gitignore` excludes secrets, `.streamlit/secrets.toml`, virtual
  environments, caches, logs, the raw CSV, downloaded archives and the row-level
  export. No secret value is printed or written to metadata.
* `pip-audit` is part of the pre-deployment check (see below).

---

## 12. Model assumptions and limitations

* The two predictors are assumed to be roughly comparably scaled after
  standardising, and K-means assumes the groups are roughly round and of similar
  size in that two-number space. That is an assumption about these numbers, not a
  fact about customer behaviour.
* K-means finds a local optimum. `random_state=42` and `n_init=20` make the run
  repeatable and reduce the chance of a poor local solution, but neither proves
  the solution is the global best.
* The groups were fitted on 200 customers. Nothing here shows they generalise.
* The dataset was created for learning, and its source terms are unclear rather
  than permissively open — see §2. That is a reason to be careful, not a legal
  conclusion.
* The spending score is assigned by the mall in the source scenario, so a change
  in how the mall assigns it would change the groups.
* Cluster descriptions are relative to this dataset's means and are regenerated
  on every training run. They are not stable marketing categories.

---

## 13. Demo walkthrough (5 minutes)

1. `python -m src.train` and read the printed report: 200 rows used, 0 excluded,
   4 shared income+score pairs kept, candidate silhouettes from k=1 to k=10, and
   the selected k with its silhouette.
2. Open `outputs/figures/elbow_curve.png` and `silhouette_by_k.png`. Point out
   that inertia never stops falling while the silhouette peaks — that is why the
   silhouette is the selection rule and the elbow is only supporting evidence.
3. Open `cluster_scatter_local_only.png`. Stars are the centroids in k$ and score
   units; the four quadrant-ish groups and the middle group are all visible. Then
   open `cluster_footprint.png` — the same story with no individual customer in
   it, which is the version that ships.
4. `python -m src.predict --income 82 --score 91` → assigned to C1, *higher
   income, higher spending score*, centroid 86.5 k$ / 82.1, distance 0.39 in
   standardised units. Now try `--income 88 --score 15` → C3, and
   `--income 26 --score 21` → C4.
5. `python -m src.predict --income 60 --score 500` → rejected with
   `That profile cannot be used: Spending score must be between 1 and 100, got
   500.` and exit code 2. Then `python -m streamlit run app.py`, press **Assign
   to nearest cluster** in the app with the same numbers, and get the same
   cluster. Try an income of 300 k$ to trigger the out-of-training-range warning.

---

## 14. Five viva questions and answers

**Q1. Why does this project use only income and spending score, and not age or
gender?**
The assignment asks to group customers by spending behaviour, and income plus
spending score are the two columns that describe it. `CustomerID` is an
identifier rather than behaviour. Age and gender are present in the file but are
demographics, and leaving them out keeps the groups defined purely by the
behaviour the task is about. Because the model never sees them, the project also
cannot — and does not — make claims about age or gender.

**Q2. Why scale the data before clustering?**
StandardScaler gives both features comparable variance so their numerical scales
do not disproportionately influence distances. It does that by subtracting each
feature's mean and dividing by its standard deviation. This matters because
K-means minimises squared distance, and squared distance weights a column by its
variance. Whether an unscaled feature would dominate depends on its actual
numerical spread, not on what its unit is called — a column labelled "k$" is not
special, and renaming it changes nothing. In this dataset the two standard
deviations are close, about 26.3 and 25.8, so scaling is a safeguard here rather
than a correction; on data whose columns are recorded on very different scales it
changes the answer substantially.

**Q3. How was the number of clusters chosen, and why not just pick 5?**
Inertia was recorded for k=1 to 10 and the silhouette score for k=2 to 10, both
measured in the same standardised matrix the model learns from. k is the
candidate with the **highest measured silhouette score**, with exact ties going
to the smaller k. Inertia always falls as k rises, so it is only useful for
spotting an elbow and it cannot select k. k=5 was not hardcoded — on this data it
happened to win on the measured score.

**Q4. Why is there no accuracy score for this model?**
Because accuracy needs ground-truth labels to compare against, and this dataset
has none. There is no "correct" segment for a customer, so there is nothing to
score a prediction against. Silhouette and inertia measure how separated and how
tight the groups are — they describe the geometry of the grouping, not its
correctness. Everything reported is descriptive, measured on the dataset the
model was fitted on; it is not held-out validation.

**Q5. If I give the model a customer with an income of 500 k$, is it confident
that this is the right group?**
No. It returns the *nearest* centroid, because that is the only answer it has, and
it says so with a warning, because 500 k$ is far outside the 15–137 k$ range it
was trained on and no customer in the file looks like that. It also does not
report a confidence percentage, because K-means gives no probabilities — the only
number it can produce is a distance, and that is a geometric distance in
standardised units, not a likelihood.

---

## 15. License and third-party notices for this repository

This project is a student assignment. The dataset file itself is **not**
redistributed here, because the source page's licence field points at a
description that contains no licence terms — see
[§2 Licensing](#licensing-what-is-verifiable-and-what-is-not) for exactly what was
and was not verified.

The dataset is credited to **Vijay Choudhary** (Kaggle user `vjchoudhary7`),
dataset id `42674`. Its own Acknowledgements read, verbatim: "From Udemy's
Machine Learning A-Z course." and "I am new to Data science field and want to
share my knowledge to others", and it links to that course's copy of
`Mall_Customers.csv`. Those notices are reproduced rather than paraphrased so the
attribution stays intact. See
[§2](#2-dataset-source-and-manual-download).
