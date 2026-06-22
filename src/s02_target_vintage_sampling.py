"""
STAGE 2 — TARGET VARIABLE CREATION (Vintage Analysis + Sampling)
==================================================================
Two industry-standard jobs happen here before any modelling starts:

1. VINTAGE ANALYSIS
   For each disbursal-quarter cohort ("vintage"), we track the cumulative
   bad rate by Month-on-Book (MOB). This tells the bank:
     - the seasoning period needed before a loan can be reliably labelled
       Good/Bad (the "outcome window")
     - whether early vintages are riskier than recent ones (macro drift)
   We pick the Outcome Period = the MOB at which the marginal bad-rate
   addition becomes negligible (curve flattens) — here observed ~12 MOB.

2. TARGET DEFINITION & SAMPLING
   - Bad  = Default flag = 1 within the observation window
   - Good = Default flag = 0, fully seasoned (>= outcome period months on book)
   - Indeterminate = not yet seasoned -> excluded from training (industry
     practice: never label a loan that hasn't had time to default)
   - Stratified Train/Test split + a true Out-of-Time (OOT) sample built
     from the most recent disbursal vintage (not random!) to mimic
     real validation conditions.
"""
import pandas as pd
import numpy as np
from config import RAW_DIR, PROC_DIR, FIG_DIR, TAB_DIR, SEED, TEST_SIZE, OOT_FRACTION

np.random.seed(SEED)


def build_vintage_curve(master, loan):
    df = master.merge(loan[["Loan_ID", "Disbursal_Date"]], on="Loan_ID", how="left")
    df["Disbursal_Date"] = pd.to_datetime(df["Disbursal_Date"])
    df["Default_Date"] = pd.to_datetime(df["Default_Date"], errors="coerce")
    df["Vintage_Q"] = df["Disbursal_Date"].dt.to_period("Q").astype(str)
    df["MOB_at_default"] = (
        (df["Default_Date"].dt.year - df["Disbursal_Date"].dt.year) * 12
        + (df["Default_Date"].dt.month - df["Disbursal_Date"].dt.month)
    )
    asof = pd.Timestamp("2023-12-31")
    df["MOB_on_book"] = (
        (asof.year - df["Disbursal_Date"].dt.year) * 12
        + (asof.month - df["Disbursal_Date"].dt.month)
    ).clip(lower=0)

    rows = []
    for vintage, grp in df.groupby("Vintage_Q"):
        n = len(grp)
        for mob in range(0, 25, 3):
            defaulted_by_mob = grp[(grp["Default"] == 1) & (grp["MOB_at_default"] <= mob)]
            rows.append({"Vintage_Q": vintage, "MOB": mob, "n_accounts": n,
                         "cum_bad_rate": round(len(defaulted_by_mob) / n * 100, 2)})
    curve = pd.DataFrame(rows)
    overall = curve.groupby("MOB")["cum_bad_rate"].mean().reset_index()
    overall["marginal_increase"] = overall["cum_bad_rate"].diff().fillna(0).round(2)
    return df, curve, overall


def plot_vintage(curve, overall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for vintage, grp in curve.groupby("Vintage_Q"):
        axes[0].plot(grp["MOB"], grp["cum_bad_rate"], marker="o", alpha=0.6, label=vintage)
    axes[0].set_title("Vintage Curves: Cumulative Bad Rate by MOB")
    axes[0].set_xlabel("Month on Book (MOB)")
    axes[0].set_ylabel("Cumulative Bad Rate (%)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].bar(overall["MOB"].astype(str), overall["marginal_increase"], color="indianred")
    axes[1].set_title("Marginal Bad-Rate Addition by MOB (curve flattens -> outcome period)")
    axes[1].set_xlabel("MOB")
    axes[1].set_ylabel("Marginal increase in bad rate (pp)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_vintage_curves.png", dpi=130)
    plt.close()


def define_target_and_sample(df, outcome_mob=12):
    df = df.copy()
    df["Target"] = np.where(
        df["Default"] == 1, 1,
        np.where(df["MOB_on_book"] >= outcome_mob, 0, np.nan)
    )
    excluded = df["Target"].isna().sum()
    modelling_df = df[df["Target"].notna()].copy()
    modelling_df["Target"] = modelling_df["Target"].astype(int)

    # True out-of-time sample: most recent disbursal vintage slice
    modelling_df = modelling_df.sort_values("Disbursal_Date")
    cutoff_idx = int(len(modelling_df) * (1 - OOT_FRACTION))
    cutoff_date = modelling_df.iloc[cutoff_idx]["Disbursal_Date"]
    oot = modelling_df[modelling_df["Disbursal_Date"] >= cutoff_date].copy()
    dev = modelling_df[modelling_df["Disbursal_Date"] < cutoff_date].copy()

    # Stratified train/test on the development sample
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        dev, test_size=TEST_SIZE, stratify=dev["Target"], random_state=SEED
    )
    for name, part in [("TRAIN", train), ("TEST", test), ("OOT", oot)]:
        print(f"{name:<6} n={len(part):<5} bad_rate={part['Target'].mean()*100:.1f}%")

    train["sample"] = "train"
    test["sample"] = "test"
    oot["sample"] = "oot"
    full = pd.concat([train, test, oot], axis=0)
    return full, excluded, outcome_mob


def main():
    master = pd.read_excel(f"{RAW_DIR}/10_master_modelling_table.xlsx", sheet_name=0)
    loan = pd.read_excel(f"{RAW_DIR}/04_loan_account_data.xlsx", sheet_name=0)

    df, curve, overall = build_vintage_curve(master, loan)
    plot_vintage(curve, overall)
    overall.to_csv(f"{TAB_DIR}/02_vintage_curve_summary.csv", index=False)
    print("Vintage curve (overall, pooled across cohorts):")
    print(overall.to_string(index=False))

    outcome_mob = 12  # chosen where marginal increase flattens
    full, excluded, outcome_mob = define_target_and_sample(df, outcome_mob)
    print(f"\nOutcome period selected: {outcome_mob} MOB")
    print(f"Indeterminate accounts excluded (not yet seasoned): {excluded}")

    full.to_csv(f"{PROC_DIR}/02_sampled_target.csv", index=False)
    print(f"\nSampled & target-tagged dataset -> {PROC_DIR}/02_sampled_target.csv "
          f"(shape={full.shape})")


if __name__ == "__main__":
    main()
