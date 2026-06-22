# Methodology Notes

Supplementary detail to the README, useful for interview prep / deep-dive questions.

## Why logistic regression on WOE, not XGBoost/Random Forest?
A PD/EWS scorecard sits inside a regulated lending decision (SR 11-7 in the US,
RBI model-risk expectations in India). Logistic regression on WOE-transformed
variables gives:
- A coefficient per variable that is directly interpretable ("for every WOE-bin
  step, odds of default change by exp(coefficient)").
- A clean mapping to scorecard points (Stage 7) that a credit committee can
  read without a data-science background.
- Stable behaviour under the kind of small, expensive-to-relabel datasets banks
  actually have for the "Bad" class (months/years to season a loan to default).
Tree ensembles (XGBoost/LightGBM/Random Forest) are commonly used as a
*challenger model* or for an *unconstrained behavioural score* feeding into the
collections/EWS triggers — see the Behavioral_Data and ML-workflow context in
the project history — but the *regulatory PD scorecard* itself defaults to
logistic regression on WOE almost universally in BFSI practice.

## Why WOE merging instead of dropping non-monotonic variables outright?
A raw equal-frequency binning on a modest sample will often show a non-monotonic
WOE pattern purely from sampling noise in individual bins. The correct response
(and what real model-dev teams do) is **coarse classing**: merge the two
adjacent bins with the smallest WOE gap, recompute, and repeat until the trend
is monotonic or business-explainable. Dropping the variable on the first
non-monotonic raw pass would throw away genuinely predictive variables.

## Why is the IV for `Bounce_Count_6M` so high (5.52) — and why drop it?
An IV this large almost always signals one of:
1. **Leakage** — the variable is observed *after* the outcome is already
   determined (e.g. bounce counts spike once a customer is already defaulting).
2. **A near-deterministic synthetic-data artefact** — common in illustrative/
   synthetic datasets where one feature was generated to strongly correlate
   with the target.
Either way, the correct action is not "great predictor, keep it" — it's
investigate, and if the causal direction is suspect or the relationship looks
too good to be true, exclude it. This project documents and drops it rather
than quietly using it, which is the responsible thing to do in a real model
build (and exactly what a Model Validation reviewer would ask about).

## Train vs Test/OOT Gini gap — what would fix it in production?
A 500-record sample is far smaller than any real bank EWS training population
(typically 50k–500k+ accounts). With only 12 variables and ~280 training rows,
the logistic regression has room to fit train-specific noise. In production,
this gap is closed by: (a) materially larger sample size, (b) tighter variable
reduction (drop borderline-IV variables that are likely noise), (c) regularised
logistic regression (L1/L2) instead of plain MLE, and (d) k-fold cross-validation
during variable selection, not just a single train/test split.

## Reject inference — when does it actually matter here?
This dataset is 100% booked/approved accounts (every Customer_ID has a loan
outcome). Reject inference matters once you build a model intended to make
*future accept/reject decisions*, because the accepts-only model is blind to
how today's rejects would have performed. The three standard techniques —
hard augmentation, parcelling, fuzzy augmentation — are implemented as a
ready-to-use utility (`reject_inference_parcelling` in `src/s06_model_creation.py`)
for the day a reject population becomes available (e.g. from a LOS extract of
declined applications).

## PSI interpretation bands used here
| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10–0.25 | Moderate shift | Monitor closely, investigate driver |
| > 0.25 | Significant shift | Recalibrate / rebuild model |

This project's Train-vs-OOT PSI = 0.077 → stable.
