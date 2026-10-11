"""统一项目路径、实验分组及固定参数。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"
SEED = 20261011
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]

# history_segment 是 history 的分箱，审计时保留，建模时省略冗余变量。
FEATURES = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
PRE_COLUMNS = FEATURES + ["history_segment"]
OUTCOMES = ["conversion", "visit", "spend"]
DEFAULT_COST = 0.02
DEFAULT_MARGIN = 30.0

def ensure_directories():
    for folder in [RAW, OUT, FIG]:
        folder.mkdir(parents=True, exist_ok=True)
