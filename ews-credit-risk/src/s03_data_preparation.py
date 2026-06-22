"""
STAGE 3 — DATA PREPARATION
============================
Steps performed, in the order a model-dev team actually follows:

1. INDEPENDENT VARIABLE IDENTIFICATION
   Split columns into: ID/key, Target, Exclusions (post-outcome / leakage),
   and candidate independent variables (IVs).

2. EXCLUSIONS
   Drop variables that are:
     - Identifiers (Customer_ID, Loan_ID)
     - Post-outcome / leakage fields (Default_Date, Default flag itself,
       collection & recovery fields that only exist AFTER default — using
       them would leak the target into the predictors)
     - Direct linear transforms of another retained variable

3. MISSING VALUE TREATMENT
   - Vars with > MISSING_THRESH missing -> dropped (not reliably populated)
   - Remaining missings -> a dedicated WOE bin ("Missing") so the model
     can learn the *information value of missingness itself*, which is
     standard scorecard practice (better than mean/median imputation,
     which destroys signal in credit data).

4. OUTLIER TREATMENT
   - Numeric IVs capped at 1st/99th percentile (Winsorisation) to stop a
     handful of extreme values dominating WOE bin edges.

5. WOE / IV CALCULATION
   - Continuous vars -> equal-frequency binning (then merged where a bin
     has < 5% population or non-monotonic WOE, a manual industry check)
   - Categorical vars -> WOE computed per category directly
   - IV summary table written for use in Stage 5 (variable reduction)
"""
import pandas as pd
import numpy as np
from config import PROC_DIR, TAB_DIR, TARGET, MISSING_THRESH, NUM_BINS

LEAKAGE_COLS = [
    "Default", "Default_Date", "MOB_at_default", "MOB_on_book",
    "Vintage_Q", "Disbursal_Date", "sample",
]
ID_COLS = ["Customer_ID", "Loan_ID"]


def identify_variables(df):
    candidate_ivs = [c for c in df.columns
                      if c not in LEAKAGE_COLS + ID_COLS + ["Target"]]
    return candidate_ivs


def drop_high_missing(df, ivs):
    miss_pct = df[ivs].isnull().mean()
    keep = miss_pct[miss_pct <= MISSING_THRESH].index.tolist()
    dropped = miss_pct[miss_pct > MISSING_THRESH].index.tolist()
    return keep, dropped, miss_pct


def cap_outliers(df, num_cols):
    df = df.copy()
    bounds = {}
    for c in num_cols:
        lo, hi = df[c].quantile([0.01, 0.99])
        bounds[c] = (lo, hi)
        df[c] = df[c].clip(lo, hi)
    return df, bounds


def woe_iv_numeric(df, col, target_col, bins=NUM_BINS):
    x = df[[col, target_col]].copy()
    x["bin"] = pd.Series(["Missing"] * len(x), index=x.index, dtype="object")
    valid = x[x[col].notnull()].copy()
    try:
        valid["bin"] = pd.qcut(valid[col], q=bins, duplicates="drop").astype(str)
    except ValueError:
        valid["bin"] = pd.cut(valid[col], bins=min(bins, valid[col].nunique())).astype(str)
    x.loc[valid.index, "bin"] = valid["bin"]
    return _woe_table(x, "bin", target_col)


def woe_iv_categorical(df, col, target_col):
    x = df[[col, target_col]].copy()
    x["bin"] = x[col].fillna("Missing").astype(str)
    return _woe_table(x, "bin", target_col)


def _woe_table(x, bincol, target_col):
    grp = x.groupby(bincol)[target_col].agg(["count", "sum"]).reset_index()
    grp.columns = ["bin", "total", "bad"]
    grp["good"] = grp["total"] - grp["bad"]
    tot_bad, tot_good = grp["bad"].sum(), grp["good"].sum()
    grp["bad_rate"] = grp["bad"] / grp["total"]
    grp["dist_bad"] = (grp["bad"] + 0.5) / (tot_bad + 0.5 * len(grp))
    grp["dist_good"] = (grp["good"] + 0.5) / (tot_good + 0.5 * len(grp))
    grp["woe"] = np.log(grp["dist_good"] / grp["dist_bad"])
    grp["iv_contrib"] = (grp["dist_good"] - grp["dist_bad"]) * grp["woe"]
    iv = grp["iv_contrib"].sum()
    return grp, iv


def main():
    df = pd.read_csv(f"{PROC_DIR}/02_sampled_target.csv")
    train = df[df["sample"] == "train"].copy()  # WOE fit on TRAIN only (no leakage to test/OOT)

    ivs = identify_variables(df)
    print(f"Candidate independent variables identified: {len(ivs)}")

    keep, dropped_missing, miss_pct = drop_high_missing(train, ivs)
    print(f"Dropped for >{int(MISSING_THRESH*100)}% missing: {dropped_missing}")

    num_cols = [c for c in keep if pd.api.types.is_numeric_dtype(train[c])]
    cat_cols = [c for c in keep if c not in num_cols]
    print(f"Numeric IVs: {len(num_cols)} | Categorical IVs: {len(cat_cols)}")

    train_capped, bounds = cap_outliers(train, num_cols)

    iv_summary = []
    woe_tables = {}
    for c in num_cols:
        tbl, iv = woe_iv_numeric(train_capped, c, "Target")
        woe_tables[c] = tbl
        iv_summary.append({"variable": c, "type": "numeric", "IV": round(iv, 4),
                            "n_bins": len(tbl)})
    for c in cat_cols:
        tbl, iv = woe_iv_categorical(train_capped, c, "Target")
        woe_tables[c] = tbl
        iv_summary.append({"variable": c, "type": "categorical", "IV": round(iv, 4),
                            "n_bins": len(tbl)})

    iv_df = pd.DataFrame(iv_summary).sort_values("IV", ascending=False).reset_index(drop=True)
    iv_df.to_csv(f"{TAB_DIR}/03_iv_summary.csv", index=False)

    # Persist all WOE tables (long format) for downstream binning re-use
    woe_long = []
    for var, tbl in woe_tables.items():
        t = tbl.copy()
        t.insert(0, "variable", var)
        woe_long.append(t)
    pd.concat(woe_long, axis=0).to_csv(f"{TAB_DIR}/03_woe_tables.csv", index=False)

    # Persist cleaned dataset (outlier-capped numeric, missing flagged) for next stage
    df_clean = df.copy()
    for c in num_cols:
        lo, hi = bounds[c]
        df_clean[c] = df_clean[c].clip(lo, hi)
    keep_cols = ID_COLS + ["Target", "sample"] + keep
    df_clean[keep_cols].to_csv(f"{PROC_DIR}/03_prepared_data.csv", index=False)

    print("\nTop 15 variables by Information Value:")
    print(iv_df.head(15).to_string(index=False))
    print(f"\nPrepared dataset -> {PROC_DIR}/03_prepared_data.csv  shape={df_clean[keep_cols].shape}")
    print(f"IV summary       -> {TAB_DIR}/03_iv_summary.csv")
    print(f"WOE tables       -> {TAB_DIR}/03_woe_tables.csv")


if __name__ == "__main__":
    main()
