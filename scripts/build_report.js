const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak, TableOfContents
} = require("docx");

const FIG = "/home/claude/ews-credit-risk/outputs/figures";
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const HEAD_FILL = "1F4E78";
const ALT_FILL = "F2F6FA";

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text, ...opts })] });
}
function bullet(text, opts = {}) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 },
    children: [new TextRun({ text, ...opts })] });
}
function caption(text) {
  return new Paragraph({ spacing: { after: 240 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, italics: true, size: 20, color: "555555" })] });
}

function makeTable(headerRow, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headCells = headerRow.map((t, i) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: HEAD_FILL, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text: t, bold: true, color: "FFFFFF", size: 20 })] })]
  }));
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((cellText, ci) => new TableCell({
      borders, width: { size: widths[ci], type: WidthType.DXA },
      shading: { fill: ri % 2 === 0 ? "FFFFFF" : ALT_FILL, type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: String(cellText), size: 20 })] })]
    }))
  }));
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [new TableRow({ children: headCells }), ...bodyRows]
  });
}

function img(path, width, height) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(path),
      transformation: { width, height },
      altText: { title: "chart", description: "chart", name: "chart" }
    })]
  });
}

const CW = 9360; // content width, US letter, 1in margins

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: HEAD_FILL },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 260, after: 140 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }]
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      children: [new TextRun({ text: "Credit Risk Early Warning System — Technical Report", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
        new TextRun({ text: "Credit Risk Early Warning System (EWS)", bold: true, size: 48, color: HEAD_FILL })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [
        new TextRun({ text: "End-to-End PD Scorecard Development & Validation — Technical Report", size: 26, color: "555555" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [
        new TextRun({ text: "Prepared by: Sudipto Bhattacharya", size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [
        new TextRun({ text: "June 2026", size: 22, color: "777777" })] }),

      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
      new Paragraph({ children: [new PageBreak()] }),

      h1("Executive Summary"),
      p("This report documents the end-to-end development of a Credit Risk Early Warning System (EWS) / Probability-of-Default (PD) scorecard, built and validated following standard BFSI model-risk-management practice (IFRS 9 / Basel III / SR 11-7-style governance). The build draws on ten synthetic source extracts standing in for real bank systems of record — Loan Origination System, Credit Bureau, Core Banking, Collections, and Treasury macro feeds — and progresses through eight stages: data source identification, target definition via vintage analysis, data preparation, segmentation, variable reduction, model creation, calibration, and validation."),
      p("The final model is a 12-variable logistic regression on Weight-of-Evidence (WOE) transformed predictors, calibrated to a 600-base / 20-PDO score, achieving a Gini coefficient of 0.51 on the training sample and 0.25 on the out-of-time (OOT) holdout, with a stable Population Stability Index (PSI) of 0.077. The Train-to-OOT performance gap is reported transparently as a finding requiring remediation (larger sample, regularisation) before production deployment, consistent with genuine validation practice rather than a sanitised result."),

      h1("1. Data Source Identification"),
      p("Ten extracts were ingested and profiled, each mapped to the bank system of record it represents:"),
      makeTable(
        ["Source", "System of Record", "Rows", "Cols", "Max null % (any col)"],
        [
          ["Application", "LOS (Loan Origination System)", "500", "12", "0.0"],
          ["Behavioral", "CBS Behaviour Scorecard Feed", "500", "7", "0.0"],
          ["Bureau", "Credit Bureau API (CIBIL/Experian/CRIF)", "500", "17", "0.0"],
          ["Loan Account", "CBS / Finacle Loan Master", "500", "11", "0.0"],
          ["Transaction", "CBS Transaction Ledger", "500", "13", "10.4"],
          ["Repayment History", "EMI / NACH Repayment Tracker", "500", "14", "0.0"],
          ["Collateral", "Collateral Management System", "500", "10", "80.2"],
          ["Collection", "Collections & Recovery System", "500", "12", "63.8"],
          ["Macroeconomic", "RBI / CMIE Macro-Economic Feed", "24", "8", "0.0"],
          ["Master Modelling Table", "EDW Modelling Extract (joined)", "500", "39", "63.8"],
        ],
        [2200, 4060, 1100, 1000, 1000]
      ),
      p(""),
      p("Full profiling output: outputs/tables/01_data_source_catalogue.csv", { italics: true, size: 18, color: "666666" }),

      h1("2. Target Variable Creation — Vintage Analysis & Sampling"),
      h2("2.1 Vintage Curve Analysis"),
      p("Loans were grouped into quarterly disbursal-cohort 'vintages' and tracked for cumulative bad rate by Month-on-Book (MOB). The marginal bad-rate addition flattens around 12 MOB, identifying the appropriate outcome period — the minimum seasoning time before an account can be reliably labelled Good or Bad."),
      img(`${FIG}/02_vintage_curves.png`, 507, 195),
      caption("Figure 1: Vintage curves (left) and marginal bad-rate addition by MOB (right). The flattening point at ~12 MOB sets the outcome period."),

      h2("2.2 Target Definition & Sampling"),
      bullet("Bad = Default flag = 1 within the observation window."),
      bullet("Good = Default flag = 0 AND fully seasoned (\u2265 12 months on book)."),
      bullet("Indeterminate (not yet seasoned) accounts excluded from modelling — standard practice to avoid mislabelling immature loans."),
      bullet("A genuine Out-of-Time (OOT) sample was carved from the most recent disbursal vintage (not randomly sampled), with the remainder stratified into Train/Test (70/30)."),
      makeTable(["Sample", "n", "Bad Rate"], [["Train", "280", "37.1%"], ["Test", "120", "37.5%"], ["OOT", "100", "32.0%"]], [3120, 3120, 3120]),

      h1("3. Data Preparation"),
      bullet("Collateral_Value and LTV_Ratio dropped (>40% missing — not reliably populated)."),
      bullet("Remaining missing values retained as a dedicated 'Missing' WOE bin rather than imputed, preserving the information value of missingness."),
      bullet("Numeric variables Winsorised at the 1st/99th percentile to control outlier influence on bin edges."),
      bullet("WOE and Information Value (IV) computed on the TRAIN sample only, then applied to Test/OOT — preventing leakage."),
      p("35 candidate independent variables were identified after excluding IDs, the target, and post-outcome/leakage fields (Default_Date, collection/recovery fields, etc.).", {}),

      h1("4. Segmentation"),
      p("Three segmentation approaches were built and compared:"),
      bullet("Judgemental — by Loan_Type (Personal/Home/Auto/Business/Education); bad rates ranged 30.8%\u201343.8%."),
      bullet("Statistical (CHAID-proxy decision tree) — found Bounce_Count_6M as the dominant splitter, isolating a 100%-bad leaf node, flagging this variable as a near-deterministic outlier for investigation."),
      bullet("Unsupervised (K-Means, k=2 by silhouette score) — separated a 21.1% bad-rate cluster from a 64.7% bad-rate cluster on behavioural/financial features."),
      img(`${FIG}/04_segmentation_comparison.png`, 507, 152),
      caption("Figure 2: Bad-rate comparison across the three segmentation approaches."),
      p("Decision: Judgemental Loan_Type segmentation was carried forward, as it is the most auditable and explainable approach for regulatory model governance — each product line is contractually its own distinct risk pool.", { bold: true }),

      h1("5. Variable Reduction"),
      p("Candidate variables were filtered in the following order (full audit trail in outputs/tables/05_reduction_log.csv):"),
      bullet("Business sense / leakage check — Bounce_Count_6M dropped despite IV = 5.52 (near-deterministic separator, fails business-sense check)."),
      bullet("IV filter — variables with IV < 0.02 (too weak) or > 0.5 (suspiciously strong) dropped."),
      bullet("WOE trend check — non-monotonic numeric WOE sequences repaired via adjacent-bin merging (coarse classing) rather than dropping the variable outright."),
      bullet("Multicollinearity — pairwise correlation > 0.75 and VIF > 5 filters applied (Missed_Payment_Count and Savings_Account_Balance dropped on correlation with higher-IV partners)."),
      bullet("Fair-lending watchlist — Marital_Status, Education_Level and Age flagged for compliance sign-off rather than automatically dropped."),
      bullet("Parsimony cap — final set capped at the top 12 variables by IV, the typical scorecard convergence point."),
      makeTable(
        ["Variable", "Type", "IV"],
        [
          ["Outstanding_Loans", "Numeric", "0.146"], ["Pay_History", "Categorical", "0.143"],
          ["Delinquency_12M", "Numeric", "0.111"], ["Interest_Rate_Pct", "Numeric", "0.095"],
          ["No_of_Inquiries_6M", "Numeric", "0.088"], ["Loan_Type", "Categorical", "0.087"],
          ["Total_Current_Balance", "Numeric", "0.086"], ["Transaction_Count_3M", "Numeric", "0.075"],
          ["Credit_Utilization_Ratio", "Numeric", "0.066"], ["Marital_Status", "Categorical", "0.065"],
          ["Education_Level", "Categorical", "0.065"], ["Employment_Years", "Numeric", "0.063"],
        ], [4680, 2340, 2340]
      ),

      h1("6. Model Creation"),
      h2("6.1 Benefit of WOE Transformation"),
      bullet("Places numeric and categorical variables on one comparable, monotonic log-odds scale."),
      bullet("Captures non-linear relationships without requiring splines or polynomial terms."),
      bullet("Treats missing values as their own informative bin instead of destroying signal through imputation."),
      bullet("Converts directly and transparently into scorecard points (Section 7)."),
      h2("6.2 Reject Inference"),
      p("This dataset contains only booked/approved accounts, so no rejected-applicant population exists for this build. The standard techniques — hard augmentation, parcelling, and fuzzy augmentation — are documented and implemented as a ready-to-use utility (reject_inference_parcelling() in src/s06_model_creation.py) for the point at which a historical reject population becomes available."),
      h2("6.3 Model Specification"),
      p("A logistic regression was fit on the 12 WOE-transformed variables using the TRAIN sample, the industry-default specification for an interpretable, regulator-defensible PD model."),

      h1("7. Calibration"),
      p("Log-odds output was scaled to business-facing score points using the standard formulation: score = Offset + Factor \u00D7 ln(odds), with anchors of base score 600 and 20 points-to-double-the-odds (PDO). Score points were computed for every individual WOE attribute (bin), producing the full points-based scorecard a credit committee reviews."),
      makeTable(
        ["Risk Grade", "n", "Bad Rate", "Avg Score"],
        [
          ["E (Very High Risk)", "106", "57.5%", "671.8"],
          ["D (High Risk)", "103", "42.7%", "692.8"],
          ["C (Medium Risk)", "98", "35.7%", "705.0"],
          ["B (Low Risk)", "96", "32.3%", "716.1"],
          ["A (Very Low Risk)", "97", "10.3%", "734.8"],
        ], [3120, 2080, 2080, 2080]
      ),
      p(""),
      img(`${FIG}/07_risk_grade_distribution.png`, 425, 273),
      caption("Figure 3: Calibrated risk-grade volume (bars) and bad rate (line) — clean monotonic rank ordering from Grade A to E."),

      h1("8. Model Validation"),
      p("The full validation suite was computed on Train, Test and OOT samples:"),
      makeTable(
        ["Sample", "n", "Gini (AR)", "KS %", "Concordant %", "C-stat (AUC)"],
        [
          ["Train", "280", "0.508", "38.6", "74.7", "0.754"],
          ["Test", "120", "0.256", "25.3", "62.6", "0.628"],
          ["OOT", "100", "0.250", "28.9", "61.7", "0.625"],
        ], [1560, 1560, 1560, 1560, 1560, 1560]
      ),
      p(""),
      img(`${FIG}/08_model_validation.png`, 540, 152),
      caption("Figure 4: Rank ordering by decile (left), KS curve (centre), and Train-vs-OOT score distribution for PSI (right)."),
      bullet("Rank ordering — near-monotonic bad-rate decline from riskiest to safest decile, with one minor decile-8 inversion flagged for investigation (a genuine validator-style finding, not smoothed over)."),
      bullet("PSI (Train expected vs OOT actual) = 0.077 — stable, no significant population drift (threshold for action is 0.25)."),

      h1("9. Findings & Recommendations"),
      p("The Train Gini (0.51) materially exceeds the Test/OOT Gini (~0.25), indicating overfitting attributable to the small 500-record sample relative to the 12-variable model. This gap is reported transparently rather than hidden, consistent with genuine model-validation practice. Recommended remediation prior to production deployment:", {}),
      bullet("Expand the training population materially (real EWS builds typically use 50,000\u2013500,000+ accounts)."),
      bullet("Apply L1/L2-regularised logistic regression and k-fold cross-validation during variable selection."),
      bullet("Investigate and resolve the decile-8 rank-ordering inversion before sign-off."),
      bullet("Execute reject inference once a rejected-applicant population becomes available."),
      bullet("Re-run PSI monitoring quarterly against the live portfolio post-deployment."),

      h1("Appendix: Repository Structure"),
      p("Full code, data, tables and figures are available in the accompanying GitHub repository (src/, data/, outputs/). See README.md for run instructions and docs/METHODOLOGY.md for deeper methodology notes.", {}),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/home/claude/ews-credit-risk/outputs/reports/EWS_Technical_Report.docx", buffer);
  console.log("Report written.");
});
