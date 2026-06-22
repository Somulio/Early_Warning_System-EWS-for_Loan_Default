"""
STAGE 8 — MODEL VALIDATION
=============================
The full suite of statistics a Model Validation / Independent Review team
checks before sign-off, computed on TRAIN, TEST and OOT:

  - Gini coefficient / Accuracy Ratio (AR) : 2*AUC - 1, discriminatory power
  - KS statistic                          : max separation of cumulative
                                              good/bad distributions
  - Concordance / Discordance %, c-statistic (=AUC) : pairwise ranking checks
  - Rank ordering                          : bad rate must increase
                                              monotonically decile-by-decile
                                              as score decreases
  - PSI (Population Stability Index)       : TRAIN (expected) vs OOT
                                              (actual) score-distribution
                                              drift check — the standard
                                              ongoing EWS monitoring metric
"""
import pandas as pd
import numpy as np
from config import PROC_DIR, TAB_DIR, FIG_DIR


def gini_ks(df, score_col="Score", target_col="Target"):
    from sklearn.metrics import roc_auc_score
    # higher score = safer, so use -score to rank by risk for AUC orientation
    auc = roc_auc_score(df[target_col], -df[score_col])
    gini = 2 * auc - 1

    d = df.sort_values(score_col)  # ascending score = riskiest first
    d["cum_bad"] = (d[target_col] == 1).cumsum() / (d[target_col] == 1).sum()
    d["cum_good"] = (d[target_col] == 0).cumsum() / (d[target_col] == 0).sum()
    ks = float((d["cum_bad"] - d["cum_good"]).abs().max() * 100)
    return gini, auc, ks, d


def concordance_discordance(df, score_col="Score", target_col="Target", n_pairs=20000, seed=42):
    rng = np.random.default_rng(seed)
    goods = df[df[target_col] == 0][score_col].values
    bads = df[df[target_col] == 1][score_col].values
    if len(goods) == 0 or len(bads) == 0:
        return np.nan, np.nan, np.nan
    gi = rng.integers(0, len(goods), n_pairs)
    bi = rng.integers(0, len(bads), n_pairs)
    g_s, b_s = goods[gi], bads[bi]
    concordant = (g_s > b_s).mean()   # good correctly scored higher than bad
    discordant = (g_s < b_s).mean()
    tied = 1 - concordant - discordant
    return concordant * 100, discordant * 100, tied * 100


def rank_ordering_table(df, score_col="Score", target_col="Target", n_deciles=10):
    d = df.copy()
    d["decile"] = pd.qcut(d[score_col].rank(method="first"), n_deciles, labels=False)
    tbl = d.groupby("decile").agg(
        n=(target_col, "count"), bad=(target_col, "sum"),
        avg_score=(score_col, "mean")
    ).reset_index()
    tbl["bad_rate"] = (tbl["bad"] / tbl["n"] * 100).round(2)
    tbl = tbl.sort_values("avg_score").reset_index(drop=True)  # low score (risky) -> high score (safe)
    tbl["decile_rank_safe_to_risky"] = range(1, len(tbl) + 1)
    is_monotonic = bool(np.all(np.diff(tbl["bad_rate"].values) <= 1e-6))
    return tbl, is_monotonic


def psi(expected, actual, bins=10):
    cuts = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    exp_binned = pd.Series(pd.cut(expected, bins=cuts))
    act_binned = pd.Series(pd.cut(actual, bins=cuts))
    exp_pct = exp_binned.value_counts(normalize=True, sort=False)
    act_pct = act_binned.value_counts(normalize=True, sort=False)
    exp_pct, act_pct = exp_pct.align(act_pct, fill_value=1e-4)
    exp_pct = exp_pct.replace(0, 1e-4)
    act_pct = act_pct.replace(0, 1e-4)
    psi_val = float(((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum())
    return psi_val


def plot_validation(deciles, train_d, oot_d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].bar(deciles["decile_rank_safe_to_risky"].astype(str), deciles["bad_rate"], color="firebrick")
    axes[0].set_title("Rank Ordering: Bad Rate by Score Decile\n(1=riskiest, 10=safest)")
    axes[0].set_xlabel("Decile"); axes[0].set_ylabel("Bad rate (%)")
    axes[0].grid(alpha=0.3)

    train_sorted = train_d.reset_index(drop=True)
    axes[1].plot(train_sorted.index, train_sorted["cum_bad"], label="Cum. Bad")
    axes[1].plot(train_sorted.index, train_sorted["cum_good"], label="Cum. Good")
    axes[1].set_title("KS Curve (TRAIN)\nMax gap = KS statistic")
    axes[1].set_xlabel("Accounts sorted by score (ascending)"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].hist(train_d["Score"], bins=20, alpha=0.6, label="TRAIN (expected)", density=True)
    axes[2].hist(oot_d["Score"], bins=20, alpha=0.6, label="OOT (actual)", density=True)
    axes[2].set_title("Score Distribution: TRAIN vs OOT\n(PSI input)")
    axes[2].set_xlabel("Score"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_model_validation.png", dpi=130)
    plt.close()


def main():
    df = pd.read_csv(f"{PROC_DIR}/07_scored_population.csv")
    results = []
    sample_data = {}
    for name in ["train", "test", "oot"]:
        part = df[df["sample"] == name]
        gini, auc, ks, d = gini_ks(part)
        conc, disc, tied = concordance_discordance(part)
        results.append({
            "sample": name.upper(), "n": len(part), "AUC_c_stat": round(auc, 4),
            "Gini_AR": round(gini, 4), "KS_pct": round(ks, 2),
            "Concordant_pct": round(conc, 2), "Discordant_pct": round(disc, 2),
            "Tied_pct": round(tied, 2),
        })
        sample_data[name] = d

    val_df = pd.DataFrame(results)
    val_df.to_csv(f"{TAB_DIR}/08_validation_metrics.csv", index=False)

    rank_tbl, is_mono = rank_ordering_table(df[df["sample"] == "train"])
    rank_tbl.to_csv(f"{TAB_DIR}/08_rank_ordering_train.csv", index=False)

    psi_val = psi(df[df["sample"] == "train"]["Score"].values, df[df["sample"] == "oot"]["Score"].values)
    psi_band = "Stable (<0.10)" if psi_val < 0.10 else ("Moderate shift (0.10-0.25) - monitor" if psi_val < 0.25 else "Significant shift (>0.25) - recalibrate")

    plot_validation(rank_tbl, sample_data["train"], sample_data["oot"])

    with open(f"{TAB_DIR}/08_psi_result.txt", "w") as f:
        f.write(f"PSI (TRAIN expected vs OOT actual) = {psi_val:.4f}\nInterpretation: {psi_band}\n")

    print("VALIDATION METRICS (TRAIN / TEST / OOT):")
    print(val_df.to_string(index=False))
    print(f"\nRank ordering monotonic (bad rate strictly non-decreasing risky->safe direction): {is_mono}")
    print(rank_tbl[["decile_rank_safe_to_risky", "n", "bad_rate", "avg_score"]].to_string(index=False))
    print(f"\nPSI (TRAIN vs OOT) = {psi_val:.4f}  -> {psi_band}")
    print(f"\nValidation metrics  -> {TAB_DIR}/08_validation_metrics.csv")
    print(f"Rank ordering table -> {TAB_DIR}/08_rank_ordering_train.csv")
    print(f"PSI result          -> {TAB_DIR}/08_psi_result.txt")
    print(f"Validation charts   -> {FIG_DIR}/08_model_validation.png")


if __name__ == "__main__":
    main()
