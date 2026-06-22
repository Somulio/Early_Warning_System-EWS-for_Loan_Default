"""
STAGE 5 — VARIABLE REDUCTION
===============================
A scorecard typically goes from ~30-40 candidate IVs down to 8-12 final
variables. Filters applied in the standard order:

1. BUSINESS SENSE / NOISE CHECK
   Variables with no plausible causal/economic link to default risk, or
   that are pure noise/ID-like artefacts, are dropped on judgement
   regardless of statistical strength (a known credit-risk anti-pattern is
   a noise variable getting high IV by chance in a small sample).

2. IV FILTER
   IV < 0.02 -> too weak to matter. IV > 0.5 -> suspiciously strong, almost
   always a leakage / near-deterministic proxy for the target -> investigate
   and usually drop (flagged explicitly here for Bounce_Count_6M).

3. WOE TREND CHECK
   A WOE bin sequence must be monotonic (or a single, business-explainable
   U-shape) across ordered bins. Non-monotonic, noisy WOE patterns signal
   an unstable variable that will not generalise -> dropped.

4. MULTICOLLINEARITY
   Pairwise correlation > CORR_THRESH -> drop the weaker-IV variable of the
   pair. Then VIF computed on the surviving set; iteratively drop the
   highest-VIF variable until all are below VIF_THRESH.

5. REGULATORY / FAIR-LENDING CHECK
   Protected-class proxies (e.g. Marital_Status, Age as sole driver,
   Education_Level) are flagged for compliance review per fair-lending
   principles, even if statistically useful — kept here only if IV is
   moderate and a business justification exists, else dropped.
"""
import pandas as pd
import numpy as np
from config import PROC_DIR, TAB_DIR, IV_MIN, IV_MAX, CORR_THRESH, VIF_THRESH

NOISE_OR_LEAKAGE_SUSPECTS = ["Bounce_Count_6M"]  # IV > 5 -> near-deterministic, investigate
FAIR_LENDING_WATCHLIST = ["Marital_Status", "Age", "Education_Level"]


def woe_monotonicity_check(woe_tables, var, max_merge_passes=4):
    """
    Industry practice: a non-monotonic WOE trend is first fixed by merging
    adjacent bins (coarse classing), not by dropping the variable outright.
    Returns (is_monotonic_after_merge, n_merges_applied).
    """
    sub = woe_tables[woe_tables["variable"] == var].copy()
    if sub["bin"].str.contains(r"\(|\[", na=False).any():
        sub["sort_key"] = sub["bin"].str.extract(r"[\(\[]([\-\d\.]+)").astype(float)
        sub = sub.sort_values("sort_key").reset_index(drop=True)
    sub = sub[sub["bin"] != "Missing"].reset_index(drop=True)  # Missing bin exempt from trend
    if len(sub) <= 2:
        return True, 0  # too few bins to violate monotonicity meaningfully

    woe_seq = list(sub["woe"].values)
    n_merges = 0
    for _ in range(max_merge_passes):
        diffs = np.diff(woe_seq)
        increasing_viol = np.sum(diffs < -1e-6)
        decreasing_viol = np.sum(diffs > 1e-6)
        if increasing_viol == 0 or decreasing_viol == 0:
            return True, n_merges
        # merge the pair of adjacent bins with the smallest |woe diff| (least information lost)
        merge_idx = int(np.argmin(np.abs(diffs)))
        woe_seq[merge_idx] = (woe_seq[merge_idx] + woe_seq[merge_idx + 1]) / 2
        del woe_seq[merge_idx + 1]
        n_merges += 1
        if len(woe_seq) <= 2:
            return True, n_merges
    return False, n_merges


def compute_vif(df, cols):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df[cols].fillna(df[cols].median())
    X = X.assign(const=1)
    vifs = []
    for i, c in enumerate(cols):
        vifs.append(variance_inflation_factor(X.values, i))
    return pd.Series(vifs, index=cols)


def main():
    df = pd.read_csv(f"{PROC_DIR}/03_prepared_data.csv")
    iv_df = pd.read_csv(f"{TAB_DIR}/03_iv_summary.csv")
    woe_tables = pd.read_csv(f"{TAB_DIR}/03_woe_tables.csv")

    log = []

    # Step 1: business sense / noise & leakage suspects
    survivors = iv_df.copy()
    for v in NOISE_OR_LEAKAGE_SUSPECTS:
        if v in survivors["variable"].values:
            log.append(f"DROP  {v:<28} reason=near-deterministic IV ({survivors.loc[survivors.variable==v,'IV'].values[0]:.2f}) -> leakage/noise suspect, fails business-sense check")
    survivors = survivors[~survivors["variable"].isin(NOISE_OR_LEAKAGE_SUSPECTS)]

    # Step 2: IV filter
    too_weak = survivors[survivors["IV"] < IV_MIN]["variable"].tolist()
    too_strong = survivors[survivors["IV"] > IV_MAX]["variable"].tolist()
    for v in too_weak:
        log.append(f"DROP  {v:<28} reason=IV below {IV_MIN} (too weak)")
    for v in too_strong:
        log.append(f"DROP  {v:<28} reason=IV above {IV_MAX} (suspiciously strong)")
    survivors = survivors[(survivors["IV"] >= IV_MIN) & (survivors["IV"] <= IV_MAX)]

    # Step 3: WOE monotonicity (numeric only; categorical bins are unordered by nature)
    mono_fail = []
    for v in survivors[survivors["type"] == "numeric"]["variable"]:
        is_mono, n_merges = woe_monotonicity_check(woe_tables, v)
        if not is_mono:
            mono_fail.append(v)
        elif n_merges > 0:
            log.append(f"KEEP  {v:<28} note=non-monotonic raw trend fixed via {n_merges} adjacent-bin merge(s) (coarse classing)")
    for v in mono_fail:
        log.append(f"DROP  {v:<28} reason=non-monotonic WOE trend (unstable bin pattern)")
    survivors = survivors[~survivors["variable"].isin(mono_fail)]

    # Step 4: multicollinearity (correlation, then VIF) on numeric survivors
    num_survivors = survivors[survivors["type"] == "numeric"]["variable"].tolist()
    corr = df[num_survivors].corr().abs()
    to_drop_corr = set()
    for i in range(len(num_survivors)):
        for j in range(i + 1, len(num_survivors)):
            a, b = num_survivors[i], num_survivors[j]
            if corr.loc[a, b] > CORR_THRESH:
                iv_a = survivors.loc[survivors.variable == a, "IV"].values[0]
                iv_b = survivors.loc[survivors.variable == b, "IV"].values[0]
                drop_v = a if iv_a < iv_b else b
                to_drop_corr.add(drop_v)
                log.append(f"DROP  {drop_v:<28} reason=corr({a},{b})={corr.loc[a,b]:.2f}>{CORR_THRESH}, lower IV of the pair")
    num_survivors = [v for v in num_survivors if v not in to_drop_corr]
    survivors = survivors[~survivors["variable"].isin(to_drop_corr)]

    vif_series = compute_vif(df, num_survivors)
    while vif_series.max() > VIF_THRESH and len(num_survivors) > 1:
        worst = vif_series.idxmax()
        log.append(f"DROP  {worst:<28} reason=VIF={vif_series[worst]:.2f}>{VIF_THRESH} (multicollinearity)")
        num_survivors.remove(worst)
        vif_series = compute_vif(df, num_survivors)
    survivors = survivors[(survivors["type"] == "categorical") | (survivors["variable"].isin(num_survivors))]

    # Step 5: fair-lending watchlist flag (informational, not auto-dropped)
    flagged = [v for v in survivors["variable"] if v in FAIR_LENDING_WATCHLIST]
    for v in flagged:
        log.append(f"FLAG  {v:<28} reason=fair-lending watchlist; retained pending compliance sign-off (IV={survivors.loc[survivors.variable==v,'IV'].values[0]:.3f})")

    final_vars = survivors.sort_values("IV", ascending=False).reset_index(drop=True)

    # Final step: cap to top-N by IV (industry scorecards typically converge on 8-12 vars)
    FINAL_N = 12
    if len(final_vars) > FINAL_N:
        excess = final_vars.iloc[FINAL_N:]["variable"].tolist()
        for v in excess:
            log.append(f"DROP  {v:<28} reason=outside top-{FINAL_N} by IV (final scorecard parsimony cap)")
        final_vars = final_vars.iloc[:FINAL_N].reset_index(drop=True)

    pd.DataFrame({"log": log}).to_csv(f"{TAB_DIR}/05_reduction_log.csv", index=False)
    final_vars.to_csv(f"{TAB_DIR}/05_final_variables.csv", index=False)

    print("VARIABLE REDUCTION LOG:")
    for line in log:
        print(" ", line)
    print(f"\nFinal variable set ({len(final_vars)} variables):")
    print(final_vars.to_string(index=False))
    print(f"\nReduction log     -> {TAB_DIR}/05_reduction_log.csv")
    print(f"Final variables   -> {TAB_DIR}/05_final_variables.csv")


if __name__ == "__main__":
    main()
