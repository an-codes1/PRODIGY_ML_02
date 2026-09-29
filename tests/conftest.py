"""Shared pytest fixtures.

Synthetic data is used **only** here, and only for automated tests. The real
project numbers come from ``data/Mall_Customers.csv``; the tests never pretend
otherwise, and the AppTest end-to-end check is skipped when the real artifacts
have not been created yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean_dataset, load_dataset
from src.train import run_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "app.py"

#: A tiny, fully synthetic stand-in for the mall file. Clearly labelled so it is
#: never mistaken for the real dataset or for real customer behaviour.
SYNTHETIC_ROWS = 60


def _synthetic_frame(n_rows: int = SYNTHETIC_ROWS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = np.round(rng.uniform(20, 130, n_rows))
    score = np.round(rng.uniform(1, 100, n_rows))
    return pd.DataFrame(
        {
            "CustomerID": np.arange(1, n_rows + 1),
            "Gender": rng.choice(["Male", "Female"], n_rows),
            "Age": rng.integers(18, 71, n_rows),
            config.INCOME_COLUMN: income,
            config.SPENDING_SCORE_COLUMN: score,
        }
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    return _synthetic_frame()


@pytest.fixture()
def synthetic_csv(tmp_path: Path, synthetic_frame: pd.DataFrame) -> Path:
    return _write_frame(synthetic_frame, tmp_path / "synthetic_customers.csv")


@pytest.fixture()
def messy_frame() -> pd.DataFrame:
    """Valid data plus one row for every cleaning rule."""
    frame = _synthetic_frame()
    extras = pd.DataFrame(
        {
            "CustomerID": [9001, 9002, 9003, 9004, 9005, 9006],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "Age": [30, 30, 30, 30, 30, 30],
            config.INCOME_COLUMN: [np.nan, "not a number", 40, 40, -5, 60],
            config.SPENDING_SCORE_COLUMN: [50, 50, np.nan, 500, 50, "42.5"],
        }
    )
    return pd.concat([frame, extras], ignore_index=True)


@pytest.fixture()
def messy_csv(tmp_path: Path, messy_frame: pd.DataFrame) -> Path:
    return _write_frame(messy_frame, tmp_path / "messy_customers.csv")


@pytest.fixture()
def trained_artifacts(tmp_path: Path, synthetic_csv: Path):
    """Train into a temporary folder so tests never touch models/ or outputs/."""
    result = run_training(
        synthetic_csv,
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        write_figures=False,
    )
    return result


@pytest.fixture()
def trained_dirs(trained_artifacts) -> tuple[Path, Path]:
    """Temporary model and output folders produced by ``trained_artifacts``."""
    root = trained_artifacts.written_files[0].parent.parent
    return root / "models", root / "outputs"


@pytest.fixture()
def real_artifacts():
    """The artifacts that ship with the project, when training has been run."""
    model_file = config.MODELS_DIR / config.MODEL_FILENAME
    metadata_file = config.MODELS_DIR / config.METADATA_FILENAME
    if not (model_file.is_file() and metadata_file.is_file()):
        pytest.skip("Real artifacts missing. Run: python -m src.train")
    from src.predict import load_artifacts

    return load_artifacts(model_file, metadata_file)


@pytest.fixture()
def real_metadata() -> dict:
    metadata_file = config.MODELS_DIR / config.METADATA_FILENAME
    if not metadata_file.is_file():
        pytest.skip("Real metadata missing. Run: python -m src.train")
    return json.loads(metadata_file.read_text(encoding="utf-8"))


@pytest.fixture()
def real_dataset() -> "pd.DataFrame":
    """The cleaned real dataset, or a skip when the CSV is not present.

    The fitted artifacts are committed, so a published clone can run most
    real-data checks without the dataset. Only the few tests that must read rows
    back out of the original file use this fixture, and they skip when the CSV is
    absent - the normal state of a clone and of the CI workflow.
    """
    if not config.DATA_FILE.is_file():
        pytest.skip(
            "Raw dataset not present (it is not redistributed). Download it as "
            "described in the README, then re-run pytest."
        )
    return clean_dataset(load_dataset(config.DATA_FILE))
