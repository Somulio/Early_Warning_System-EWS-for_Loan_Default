"""
Credit Risk Early Warning System (EWS) — End-to-End Pipeline Orchestrator
===========================================================================
Runs all 8 modelling stages in sequence. Each stage is also independently
runnable (python src/s0X_*.py) for debugging / iterative development.

Usage:
    python src/main.py
"""
import subprocess
import sys
import time

STAGES = [
    ("s01_data_sources.py", "1. Data Source Identification"),
    ("s02_target_vintage_sampling.py", "2. Target Variable Creation (Vintage Analysis + Sampling)"),
    ("s03_data_preparation.py", "3. Data Preparation (Missing/Outlier/WOE/IV)"),
    ("s04_segmentation.py", "4. Segmentation (Judgemental/Statistical/K-Means)"),
    ("s05_variable_reduction.py", "5. Variable Reduction (IV/Correlation/VIF/Regulatory)"),
    ("s06_model_creation.py", "6. Model Creation (WOE Logistic Regression + Reject Inference)"),
    ("s07_calibration.py", "7. Calibration (Log-Odds to Score Points)"),
    ("s08_model_validation.py", "8. Model Validation (Gini/KS/PSI/Concordance/Rank Ordering)"),
]


def main():
    t0 = time.time()
    for script, title in STAGES:
        print(f"\n{'='*70}\nSTAGE {title}\n{'='*70}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\n[FAILED] {script} exited with code {result.returncode}. Stopping pipeline.")
            sys.exit(1)
    print(f"\n{'='*70}\nPIPELINE COMPLETE in {time.time()-t0:.1f}s — see outputs/ for all artefacts.\n{'='*70}")


if __name__ == "__main__":
    main()
