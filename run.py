"""按阶段执行：python run.py --stage experiment，或 --stage all。"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from pathlib import Path
_plot_cache = Path(__file__).resolve().parent / ".cache" / "matplotlib"
_plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache))

from src.data import download, audit
from src.statistics import balance, ab_analysis, power_analysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["download", "audit", "experiment", "causal", "models", "report", "all"],
        default="all",
    )
    stage = parser.parse_args().stage
    if stage == "report":
        from src.reporting import create_report
        create_report()
        return
    df = download()
    print(f"DATA: {df.shape[0]:,} customers, {df.shape[1]} fields", flush=True)
    print("AUDIT", json.dumps(audit(df)), flush=True)
    if stage in ["download", "audit"]:
        return
    validation = balance(df)
    print("VALIDATION", json.dumps(validation), flush=True)
    if validation["srm_flag"]:
        raise ValueError("SRM warning: investigate assignment before causal analysis.")
    if stage in ["experiment", "all"]:
        result = ab_analysis(df)
        power_analysis(df)
        print(result[[
            "treatment", "control", "metric", "effect", "ci_low", "ci_high", "p_holm"
        ]].to_string(index=False), flush=True)
    if stage in ["causal", "all"]:
        from src.causal import causal_analysis
        causal_analysis(df)
    if stage in ["models", "all"]:
        from src.modeling import run_models
        run_models(df)
        from src.budget import budget_analysis
        budget_analysis()
    if stage == "all":
        from src.reporting import create_report
        create_report()

if __name__ == "__main__":
    main()
