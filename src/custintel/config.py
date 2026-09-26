"""Project-wide constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"  # small committed files the app reads
FIGURES_DIR = ROOT / "reports" / "figures"

DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
XLSX_NAME = "online_retail_II.xlsx"

# Prediction setup: at each snapshot date we look at the customer's past and predict the next
# HORIZON_DAYS. "Churn" = an active customer makes no purchase in that window.
HORIZON_DAYS = 180
MIN_HISTORY_DAYS = 0  # customers must have >= 1 purchase before the snapshot
TEST_SNAPSHOT = "2011-06-09"  # labels run to 2011-12-09, the last day of data. Touched once, at the end.
SCORING_SNAPSHOT = "2011-12-09"  # today's scores: labels are unknown, only predictions
# Monthly training snapshots; every label window ends on/before the test snapshot.
TRAIN_SNAPSHOTS = ["2010-03-09", "2010-04-09", "2010-05-09", "2010-06-09", "2010-07-09",
                   "2010-08-09", "2010-09-09", "2010-10-09", "2010-11-09", "2010-12-09"]
# Model-selection protocol (kept apart from the test snapshot): fit on snapshots whose label
# windows end before VALID_SNAPSHOT, then score VALID_SNAPSHOT (labels: 2010-12-09 -> 2011-06-09).
VALID_SNAPSHOT = "2010-12-09"
VALID_TRAIN_SNAPSHOTS = ["2010-03-09", "2010-04-09", "2010-05-09", "2010-06-09"]
SEED = 42

# Stock codes that are not products (fees, postage, manual adjustments, samples, ...).
NON_PRODUCT_CODES = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK", "S", "AMAZONFEE",
                     "B", "ADJUST", "ADJUST2", "TEST001", "TEST002", "GIFT"}
