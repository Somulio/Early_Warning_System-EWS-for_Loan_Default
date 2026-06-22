"""
Global configuration for the Credit Risk Early Warning System (EWS) pipeline.
All paths, constants and modelling parameters are centralised here so every
stage script (01_... to 08_...) stays short and reproducible.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
TAB_DIR = os.path.join(ROOT, "outputs", "tables")
REP_DIR = os.path.join(ROOT, "outputs", "reports")

for d in [PROC_DIR, FIG_DIR, TAB_DIR, REP_DIR]:
    os.makedirs(d, exist_ok=True)

RAW_FILES = {
    "application": "01_customer_application_data.xlsx",
    "behavioral": "02_behavioral_data.xlsx",
    "bureau": "03_bureau_data.xlsx",
    "loan": "04_loan_account_data.xlsx",
    "transaction": "05_transaction_data.xlsx",
    "repayment": "06_repayment_history_data.xlsx",
    "collateral": "07_collateral_data.xlsx",
    "collection": "08_collection_data.xlsx",
    "macro": "09_macroeconomic_data.xlsx",
    "master": "10_master_modelling_table.xlsx",
}

SEED = 42
TARGET = "Default"
ID_COLS = ["Customer_ID", "Loan_ID"]
DATE_COLS = ["Default_Date"]

# Variable reduction thresholds (industry-standard rules of thumb)
IV_MIN, IV_MAX = 0.02, 0.5         # below = weak, above = suspicious/over-fit
MISSING_THRESH = 0.40              # drop independent vars with >40% missing
CORR_THRESH = 0.75                 # drop one of a pair above this corr
VIF_THRESH = 5.0                   # multicollinearity cutoff
NUM_BINS = 5                       # default WOE bins for continuous vars

# Train/Test/OOT split
TEST_SIZE = 0.30
OOT_FRACTION = 0.20                 # most recent vintage slice held out

# EWS risk grade cut-offs on score points (after calibration)
SCORE_BASE = 600
SCORE_PDO = 20      # points to double the odds
SCORE_ODDS = 1 / 19  # odds of good:bad at base score (≈ 5% bad rate)
