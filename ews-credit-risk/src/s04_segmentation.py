"""
STAGE 4 — SEGMENTATION
=========================
Why segment at all: a single pooled scorecard often under-fits sub-populations
with structurally different risk drivers (e.g. salaried vs self-employed,
secured vs unsecured). Three industry approaches are demonstrated:

1. JUDGEMENTAL SEGMENTATION
   Business-rule driven, based on domain knowledge — here: Loan_Type
   (Personal/Home/Auto/Business/Education) since product risk drivers differ
   structurally (collateral, ticket size, tenure).

2. STATISTICAL SEGMENTATION (CHAID-style)
   A shallow decision tree on the target finds the single best splitting
   variable + cut-point that maximises bad-rate separation between
   resulting nodes — proxy for a CHAID/CART segmentation scan.

3. UNSUPERVISED SEGMENTATION (K-MEANS)
   Clusters customers purely on behavioural/financial distance (unsupervised,
   no target leakage) to discover natural risk-relevant groupings that
   business rules might miss. Optimal k chosen via silhouette score.

Each method's segment-level bad rates are compared; the report records
which segmentation is carried forward for the modelling stage (here:
judgemental Loan_Type, since each product line is contractually itself a
distinct risk pool — also the most explainable/auditable approach for a
regulator, the deciding factor cited in the validation report).
"""
import pandas as pd
import numpy as np
from config import PROC_DIR, FIG_DIR, TAB_DIR, SEED

np.random.seed(SEED)


def judgemental_segmentation(df):
    seg = df.groupby("Loan_Type")["Target"].agg(["count", "mean"]).reset_index()
    seg.columns = ["segment", "n", "bad_rate"]
    seg["method"] = "Judgemental (Loan_Type)"
    return seg


def statistical_segmentation(df):
    from sklearn.tree import DecisionTreeClassifier
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["Target"], errors="ignore")
    num_df = num_df.fillna(num_df.median())
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=40, random_state=SEED)
    tree.fit(num_df, df["Target"])
    leaf = tree.apply(num_df)
    tmp = df.copy()
    tmp["leaf"] = leaf
    seg = tmp.groupby("leaf")["Target"].agg(["count", "mean"]).reset_index()
    seg.columns = ["segment", "n", "bad_rate"]
    seg["segment"] = "Tree_Node_" + seg["segment"].astype(str)
    seg["method"] = "Statistical (CHAID-proxy decision tree)"
    top_split = num_df.columns[np.argmax(tree.feature_importances_)]
    return seg, top_split


def kmeans_segmentation(df, k_range=range(2, 6)):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    feats = ["Income_INR", "Credit_Utilization_Ratio", "Bounce_Count_6M",
              "Delinquency_12M", "Behavior_Repayment_Score", "Avg_Balance_6M"]
    X = df[feats].fillna(df[feats].median())
    Xs = StandardScaler().fit_transform(X)

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
        scores[k] = silhouette_score(Xs, km.labels_)
    best_k = max(scores, key=scores.get)

    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(Xs)
    tmp = df.copy()
    tmp["cluster"] = km.labels_
    seg = tmp.groupby("cluster")["Target"].agg(["count", "mean"]).reset_index()
    seg.columns = ["segment", "n", "bad_rate"]
    seg["segment"] = "Cluster_" + seg["segment"].astype(str)
    seg["method"] = f"K-Means (k={best_k}, silhouette={scores[best_k]:.3f})"
    return seg, scores, best_k, feats


def plot_segments(judg, stat, km):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, seg, title in zip(axes, [judg, stat, km],
                                ["Judgemental: Loan_Type", "Statistical: Tree Nodes", "Unsupervised: K-Means"]):
        ax.bar(seg["segment"].astype(str), seg["bad_rate"] * 100, color="steelblue")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Bad rate (%)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_segmentation_comparison.png", dpi=130)
    plt.close()


def main():
    df = pd.read_csv(f"{PROC_DIR}/03_prepared_data.csv")

    judg = judgemental_segmentation(df)
    stat, top_split = statistical_segmentation(df)
    km, sil_scores, best_k, feats = kmeans_segmentation(df)

    plot_segments(judg, stat, km)

    all_seg = pd.concat([judg, stat, km], axis=0)
    all_seg.to_csv(f"{TAB_DIR}/04_segmentation_summary.csv", index=False)

    print("JUDGEMENTAL SEGMENTATION (Loan_Type):")
    print(judg.to_string(index=False))
    print(f"\nSTATISTICAL SEGMENTATION (top split variable: {top_split}):")
    print(stat.to_string(index=False))
    print(f"\nK-MEANS SEGMENTATION (silhouette by k: {sil_scores}, chosen k={best_k}, features={feats}):")
    print(km.to_string(index=False))
    print(f"\nDecision: Judgemental Loan_Type segmentation carried forward — "
          f"most auditable/explainable for SR 11-7 / regulatory model governance.")
    print(f"\nSegmentation summary -> {TAB_DIR}/04_segmentation_summary.csv")
    print(f"Comparison chart      -> {FIG_DIR}/04_segmentation_comparison.png")


if __name__ == "__main__":
    main()
