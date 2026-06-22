"""
STAGE 7 — CALIBRATION (Scaling Log-Odds to Score Points)
============================================================
Converts the logistic regression's log-odds output into a business-facing
score (industry convention, same as FICO/CIBIL-style scaling):

  score = Offset + Factor * ln(odds_of_good)

Two anchors are chosen up front (standard practice):
  - SCORE_BASE  : the score at a chosen baseline odds
  - SCORE_PDO   : Points to Double the Odds (how many extra points needed
                  to double the good:bad odds)

From these:
  Factor = SCORE_PDO / ln(2)
  Offset = SCORE_BASE - Factor * ln(SCORE_ODDS)

Because the model coefficients are already on the WOE scale, this scaling
distributes cleanly down to EVERY individual WOE attribute (bin), producing
a fully transparent points-per-bin scorecard — the artefact that a credit
committee actually signs off on, not just the underlying regression.

  points_per_bin = -(woe_bin * coefficient + intercept/n_vars) * Factor
                    (intercept distributed evenly across variables, a
                    common convention so the scorecard reads positively)
"""
import pandas as pd
import numpy as np
import pickle
from config import PROC_DIR, TAB_DIR, FIG_DIR, SCORE_BASE, SCORE_PDO, SCORE_ODDS


def main():
    with open(f"{PROC_DIR}/06_model.pkl", "rb") as f:
        art = pickle.load(f)
    model, woe_maps, woe_cols = art["model"], art["woe_maps"], art["woe_cols"]

    factor = SCORE_PDO / np.log(2)
    offset = SCORE_BASE - factor * np.log(SCORE_ODDS)
    print(f"Calibration anchors: base={SCORE_BASE}, PDO={SCORE_PDO}, base_odds={SCORE_ODDS:.4f}")
    print(f"Derived Factor={factor:.4f}, Offset={offset:.4f}")

    coef_df = pd.read_csv(f"{TAB_DIR}/06_model_coefficients.csv")
    n_vars = len(woe_cols)
    intercept = coef_df["intercept"].iloc[0]

    score_rows = []
    for _, row in coef_df.iterrows():
        var_woe_col = row["variable"]
        var = var_woe_col.replace("_woe", "")
        coef = row["coefficient"]
        mapping = woe_maps[var]
        for bin_label, woe_val in mapping["woe_map"].items():
            # Target=1 is BAD; good:bad odds use -log-odds(bad) = log-odds(good)
            points = -(coef * woe_val + intercept / n_vars) * factor
            score_rows.append({
                "variable": var, "bin": bin_label, "woe": round(woe_val, 4),
                "coefficient": round(coef, 4), "score_points": round(points, 1)
            })

    score_card = pd.DataFrame(score_rows)
    score_card.to_csv(f"{TAB_DIR}/07_scorecard_points.csv", index=False)

    # Apply to full WOE-transformed dataset to get a final applicant score
    woe_df = pd.read_csv(f"{PROC_DIR}/06_woe_transformed.csv")
    scores = pd.Series(offset, index=woe_df.index)
    for col in woe_cols:
        scores += -(coef_df.loc[coef_df.variable == col, "coefficient"].values[0] * woe_df[col]
                     + intercept / n_vars) * factor
    woe_df["Score"] = scores.round(0)

    # Risk grades: quantile-based cut-offs on the realised score distribution
    # (industry practice — bands are calibrated to the portfolio's own score
    # spread at each refresh, not fixed arbitrarily, so each grade carries a
    # meaningful, non-empty population)
    q = woe_df["Score"].quantile([0.2, 0.4, 0.6, 0.8]).values
    bins = [-np.inf, q[0], q[1], q[2], q[3], np.inf]
    labels = ["E (Very High Risk)", "D (High Risk)", "C (Medium Risk)", "B (Low Risk)", "A (Very Low Risk)"]
    woe_df["Risk_Grade"] = pd.cut(woe_df["Score"], bins=bins, labels=labels)
    print(f"Quantile-based score cut-offs: {[round(x,0) for x in q]}")

    woe_df[["Target", "sample", "Score", "Risk_Grade"]].to_csv(f"{PROC_DIR}/07_scored_population.csv", index=False)

    grade_summary = woe_df.groupby("Risk_Grade", observed=True).agg(
        n=("Target", "count"), bad_rate=("Target", "mean"),
        avg_score=("Score", "mean")
    ).reset_index()
    grade_summary.to_csv(f"{TAB_DIR}/07_risk_grade_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax2 = ax.twinx()
    ax.bar(grade_summary["Risk_Grade"].astype(str), grade_summary["n"], color="lightsteelblue", label="Volume")
    ax2.plot(grade_summary["Risk_Grade"].astype(str), grade_summary["bad_rate"] * 100,
              color="darkred", marker="o", linewidth=2, label="Bad rate %")
    ax.set_ylabel("Account volume")
    ax2.set_ylabel("Bad rate (%)")
    ax.set_title("Calibrated Risk Grades: Volume vs Bad Rate")
    plt.xticks(rotation=20, fontsize=8)
    fig.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_risk_grade_distribution.png", dpi=130)
    plt.close()

    print(f"\nScorecard (points per WOE attribute) -> {TAB_DIR}/07_scorecard_points.csv")
    print(f"Scored population                     -> {PROC_DIR}/07_scored_population.csv")
    print("\nRisk grade summary:")
    print(grade_summary.to_string(index=False))
    print("\nSample of scorecard points:")
    print(score_card.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
