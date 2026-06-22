"""
STAGE 1 — DATA SOURCE IDENTIFICATION
=====================================
Industry context: an EWS model draws from multiple systems of record inside
a bank. This script (a) ingests every source file, (b) profiles it
(rows, columns, dtypes, null %, key cardinality), and (c) writes a
'Data Source Catalogue' — the artefact a Model Validation / MRM team always
asks for first — to outputs/tables/01_data_source_catalogue.csv.

Typical real-world system-of-record mapping (documented for context, since
the source files here are extracts that stand in for these systems):
  Application data   -> LOS (Loan Origination System)
  Behavioral data     -> Core Banking System (CBS) behaviour scorecard feed
  Bureau data         -> CIBIL / Experian / CRIF Highmark API
  Loan account data   -> CBS / Finacle loan master
  Transaction data    -> CBS savings/current account transaction ledger
  Repayment history   -> EMI/NACH repayment tracker
  Collateral data     -> Collateral Management System (CMS)
  Collection data      -> Collections/Recovery Management System
  Macroeconomic data  -> RBI / CMIE / internal Treasury research feed
  Master table        -> Enterprise Data Warehouse (EDW) modelling extract
"""
import pandas as pd
from config import RAW_DIR, RAW_FILES, TAB_DIR

SYSTEM_OF_RECORD = {
    "application": "LOS (Loan Origination System)",
    "behavioral": "CBS Behaviour Scorecard Feed",
    "bureau": "Credit Bureau API (CIBIL/Experian/CRIF)",
    "loan": "CBS / Finacle Loan Master",
    "transaction": "CBS Transaction Ledger",
    "repayment": "EMI / NACH Repayment Tracker",
    "collateral": "Collateral Management System",
    "collection": "Collections & Recovery System",
    "macro": "RBI / CMIE Macro-Economic Feed",
    "master": "EDW Modelling Extract (joined)",
}


def profile_source(name: str, path: str) -> dict:
    df = pd.read_excel(path, sheet_name=0)
    null_pct = (df.isnull().mean() * 100).round(2)
    key_col = "Customer_ID" if "Customer_ID" in df.columns else df.columns[0]
    return {
        "source_name": name,
        "system_of_record": SYSTEM_OF_RECORD[name],
        "file": RAW_FILES[name],
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "n_unique_keys": df[key_col].nunique(),
        "max_null_pct_any_col": float(null_pct.max()),
        "cols_with_nulls": int((null_pct > 0).sum()),
    }


def main():
    catalogue = []
    for name, fname in RAW_FILES.items():
        path = f"{RAW_DIR}/{fname}"
        catalogue.append(profile_source(name, path))
        print(f"[OK] {name:<12} -> {SYSTEM_OF_RECORD[name]}")

    cat_df = pd.DataFrame(catalogue)
    out_path = f"{TAB_DIR}/01_data_source_catalogue.csv"
    cat_df.to_csv(out_path, index=False)
    print(f"\nData source catalogue written -> {out_path}")
    print(cat_df.to_string(index=False))


if __name__ == "__main__":
    main()
