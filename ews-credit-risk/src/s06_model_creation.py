"""
STAGE 6 — MODEL CREATION
===========================
1. WOE TRANSFORMATION (benefit of WOE)
   Each final IV is replaced by its WOE value (fit on TRAIN bins only, then
   applied to TEST/OOT — no leakage). Benefits, stated explicitly because
   interviewers always probe this:
     - Puts all variables (numeric & categorical) on the same monotonic,
       log-odds scale -> coefficients are directly comparable & interpretable
     - Captures non-linear relationships without needing splines/polynomials
     - Naturally handles missing values as their own informative bin
     - Makes the final logistic regression coefficients easy to convert to
       scorecard points (Stage 7) and easy to explain to a regulator/credit
       committee ("higher WOE = safer bucket = more points")

2. REJECT INFERENCE (discussed, not literally needed on this dataset since
   it has no rejected-applicant population — but documented because every
   real bank EWS/scorecard build must address it):
   When a model is trained only on *approved* accounts, it inherits
   "approval bias" — the population of rejects is structurally different
   and unobserved as Good/Bad. Standard techniques:
     - Hard/simple augmentation: score rejects with the accepts-only model,
       assign Bad if score below cut-off, then re-train on accepts+rejects
     - Parcelling: bucket rejects by score band, assign bad-rate from the
       matching accept band's actual bad rate (probabilistic assignment)
     - Fuzzy augmentation: assign each reject a fractional good/bad weight
       equal to the model's predicted PD, duplicating the row as both
       good & bad with those weights
   This synthetic dataset only contains booked/approved loans, so no
   reject-inference step is executed — the function below is provided as
   a ready-to-use utility once a reject population exists.

3. MODEL TRAINING
   Logistic regression on WOE-transformed final variables (the industry
   default for an EWS/PD scorecard given regulatory interpretability
   requirements under SR 11-7 / RBI guidance), fit on TRAIN, evaluated on
   TEST and OOT.
"""
import pandas as pd
import numpy as np
from config import PROC_DIR, TAB_DIR, REP_DIR, SEED


def fit_woe_map(train_df, var, var_type, bins=5):
    x = train_df[[var, "Target"]].copy()
    if var_type == "numeric":
        x["bin"] = pd.Series(["Missing"] * len(x), index=x.index, dtype="object")
        valid = x[x[var].notnull()]
        try:
            cats, edges = pd.qcut(valid[var], q=bins, duplicates="drop", retbins=True)
        except ValueError:
            cats, edges = pd.cut(valid[var], bins=min(bins, valid[var].nunique()), retbins=True)
        x.loc[valid.index, "bin"] = cats.astype(str).values
    else:
        x["bin"] = x[var].fillna("Missing").astype(str)
        edges = None

    grp = x.groupby("bin")["Target"].agg(["count", "sum"]).reset_index()
    grp.columns = ["bin", "total", "bad"]
    grp["good"] = grp["total"] - grp["bad"]
    tot_bad, tot_good = grp["bad"].sum(), grp["good"].sum()
    grp["dist_bad"] = (grp["bad"] + 0.5) / (tot_bad + 0.5 * len(grp))
    grp["dist_good"] = (grp["good"] + 0.5) / (tot_good + 0.5 * len(grp))
    grp["woe"] = np.log(grp["dist_good"] / grp["dist_bad"])
    woe_map = dict(zip(grp["bin"], grp["woe"]))
    return {"type": var_type, "edges": edges, "woe_map": woe_map,
            "default_woe": 0.0}  # unseen category -> neutral WOE


def apply_woe(df, var, mapping):
    if mapping["type"] == "numeric":
        out = pd.Series(["Missing"] * len(df), index=df.index, dtype="object")
        valid = df[var].notnull()
        if mapping["edges"] is not None:
            edges = mapping["edges"].copy()
            edges[0], edges[-1] = -np.inf, np.inf
            binned = pd.cut(df.loc[valid, var], bins=edges)
            out.loc[valid] = binned.astype(str).values
    else:
        out = df[var].fillna("Missing").astype(str)
    return out.map(mapping["woe_map"]).fillna(mapping["default_woe"])


def reject_inference_parcelling(accepts_scored, rejects_scored, score_col="pd_score", n_bands=10):
    """
    Utility for when a reject population becomes available. Buckets both
    populations into the same score bands and assigns each reject record
    the *actual* bad rate observed in accepts within its band (parcelling).
    Returns rejects_scored with an added 'inferred_bad_rate' column.
    """
    accepts_scored = accepts_scored.copy()
    accepts_scored["band"] = pd.qcut(accepts_scored[score_col], q=n_bands, duplicates="drop")
    band_bad_rate = accepts_scored.groupby("band")["Target"].mean()
    edges = [interval.left for interval in band_bad_rate.index] + [band_bad_rate.index[-1].right]
    edges[0], edges[-1] = -np.inf, np.inf
    rejects_scored = rejects_scored.copy()
    rejects_scored["band"] = pd.cut(rejects_scored[score_col], bins=edges)
    band_map = {interval: rate for interval, rate in zip(band_bad_rate.index, band_bad_rate.values)}
    rejects_scored["inferred_bad_rate"] = rejects_scored["band"].map(
        lambda b: band_map.get(b, band_bad_rate.mean())
    )
    return rejects_scored


def main():
    df = pd.read_csv(f"{PROC_DIR}/03_prepared_data.csv")
    final_vars = pd.read_csv(f"{TAB_DIR}/05_final_variables.csv")
    train = df[df["sample"] == "train"].copy()
    test = df[df["sample"] == "test"].copy()
    oot = df[df["sample"] == "oot"].copy()

    woe_maps = {}
    for _, row in final_vars.iterrows():
        var, vtype = row["variable"], row["type"]
        woe_maps[var] = fit_woe_map(train, var, vtype)

    def transform(part):
        out = pd.DataFrame(index=part.index)
        for var in woe_maps:
            out[f"{var}_woe"] = apply_woe(part, var, woe_maps[var])
        out["Target"] = part["Target"].values
        return out

    train_w, test_w, oot_w = transform(train), transform(test), transform(oot)
    woe_cols = [c for c in train_w.columns if c != "Target"]

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=SEED, max_iter=1000)
    model.fit(train_w[woe_cols], train_w["Target"])

    coef_df = pd.DataFrame({"variable": woe_cols, "coefficient": model.coef_[0]})
    coef_df["intercept"] = model.intercept_[0]
    coef_df.to_csv(f"{TAB_DIR}/06_model_coefficients.csv", index=False)

    # quick in-sample sanity check (full validation happens in Stage 8)
    from sklearn.metrics import roc_auc_score
    for name, w in [("TRAIN", train_w), ("TEST", test_w), ("OOT", oot_w)]:
        pred = model.predict_proba(w[woe_cols])[:, 1]
        auc = roc_auc_score(w["Target"], pred)
        print(f"{name:<6} AUC={auc:.4f}  Gini={2*auc-1:.4f}")

    # persist WOE-transformed datasets + model for Stage 7/8 reuse
    train_w["sample"], test_w["sample"], oot_w["sample"] = "train", "test", "oot"
    full_w = pd.concat([train_w, test_w, oot_w], axis=0)
    full_w.to_csv(f"{PROC_DIR}/06_woe_transformed.csv", index=False)

    import pickle
    with open(f"{PROC_DIR}/06_model.pkl", "wb") as f:
        pickle.dump({"model": model, "woe_maps": woe_maps, "woe_cols": woe_cols}, f)

    print(f"\nModel coefficients     -> {TAB_DIR}/06_model_coefficients.csv")
    print(f"WOE-transformed data   -> {PROC_DIR}/06_woe_transformed.csv")
    print(f"Trained model artefact -> {PROC_DIR}/06_model.pkl")
    print("\nCoefficients:")
    print(coef_df.to_string(index=False))
    print("\nNote on reject inference: this dataset contains only booked/approved "
          "accounts, so no reject population exists -> reject inference is "
          "documented (see module docstring + reject_inference_parcelling()) "
          "but not executed. In a live deployment, score the historical "
          "reject population with this model and apply parcelling before "
          "the next model refresh.")


if __name__ == "__main__":
    main()
