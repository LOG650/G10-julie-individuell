from __future__ import annotations

import os
import re
import tempfile
import warnings
from pathlib import Path

import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "g10-julie-individuell-matplotlib"),
)

import matplotlib
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

matplotlib.use("Agg")

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"statsmodels\.tsa\.holtwinters\.model",
)
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"statsmodels\..*",
)
warnings.filterwarnings(
    "ignore",
    category=ValueWarning,
    module=r"statsmodels\..*",
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "004 data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELING_DIR = Path(__file__).resolve().parent
MODELING_OUTPUT_DIR = MODELING_DIR / "outputs"

MASTER_DATA_PATH = RAW_DATA_DIR / "Data som skal brukes Anonymisert.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
RESULTS_DIR = MODELING_OUTPUT_DIR / "shared"
MODEL_LOG_DIR = MODELING_OUTPUT_DIR / "logs"
MODEL_ARTIFACTS_DIR = MODELING_OUTPUT_DIR / "models"
COMPARISON_LOG_PATH = MODEL_LOG_DIR / "Modellsammenligning.md"
TRAIN_END_DATE = pd.Timestamp(year=2024, month=12, day=1)
TEST_START_DATE = pd.Timestamp(year=2025, month=1, day=1)
MAX_OFFHIRE_VALUE = 100.0

MONTH_ORDER = {
    "Januar": 1,
    "Februar": 2,
    "Mars": 3,
    "April": 4,
    "Mai": 5,
    "Juni": 6,
    "Juli": 7,
    "August": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Desember": 12,
}
MONTH_COLUMNS = list(MONTH_ORDER.keys())
MONTH_NAME_BY_NUMBER = {value: key for key, value in MONTH_ORDER.items()}
YEAR_PATTERN = re.compile(r"År:\s*(\d{4})")

MODEL_METADATA = {
    "sarima": {
        "display_name": "SARIMA",
        "function_name": "run_sarima",
    },
    "exponential_smoothing": {
        "display_name": "Eksponentiell glatting",
        "function_name": "run_exponential_smoothing",
    },
    "xgboost": {
        "display_name": "XGBoost",
        "function_name": "run_xgboost",
    },
    "lstm": {
        "display_name": "LSTM",
        "function_name": "run_lstm",
    },
}
ACTIVE_MODELS = ["sarima", "exponential_smoothing", "xgboost", "lstm"]
GENERATE_FUTURE_FORECASTS = True
FUTURE_FORECAST_HORIZONS = [1, 3, 6, 12]
MAX_FUTURE_FORECAST_HORIZON = max(FUTURE_FORECAST_HORIZONS)
