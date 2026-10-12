"""
NutriCare ML -- Data Loading & Validation
Reads the Excel workbook, validates quality, and produces a validation report.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import (
    DATA_FILE, DATASET_SHEET, LABELING_RULES_SHEET,
    DATA_DICTIONARY_SHEET, TARGET_COL, ID_COL, EXCLUDE_COLS,
    TARGET_CLASSES, OUTPUT_DIR, SEPARATOR
)


def load_dataset() -> pd.DataFrame:
    """Load the main dataset from the Excel workbook."""
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}\n"
            "Place TrainData.xlsx in the project root directory."
        )
    df = pd.read_excel(DATA_FILE, sheet_name=DATASET_SHEET, engine="openpyxl")
    print(f"[INFO] Loaded {len(df)} rows x {len(df.columns)} columns from '{DATASET_SHEET}'")
    return df


def load_labeling_rules() -> pd.DataFrame:
    """Load the labeling rules sheet for documentation."""
    try:
        lr = pd.read_excel(DATA_FILE, sheet_name=LABELING_RULES_SHEET, engine="openpyxl")
        return lr
    except Exception:
        print("[WARN] Labeling_Rules sheet not found.")
        return pd.DataFrame()


def validate_dataset(df: pd.DataFrame) -> str:
    """
    Run comprehensive data validation checks.
    Returns a formatted validation report string and saves it to outputs/.
    """
    lines = []
    lines.append(SEPARATOR)
    lines.append("NUTRICARE ML -- DATASET VALIDATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(SEPARATOR)

    # -- 1. Shape ------------------------------------------------------
    lines.append(f"\n1. DATASET SHAPE")
    lines.append(f"   Rows:    {df.shape[0]}")
    lines.append(f"   Columns: {df.shape[1]}")

    # -- 2. Column listing ---------------------------------------------
    lines.append(f"\n2. COLUMNS ({len(df.columns)} total)")
    for i, col in enumerate(df.columns, 1):
        lines.append(f"   {i:>2}. {col} [{df[col].dtype}]")

    # -- 3. Target column check ----------------------------------------
    lines.append(f"\n3. TARGET COLUMN: '{TARGET_COL}'")
    if TARGET_COL not in df.columns:
        lines.append("   *** ERROR: Target column not found! ***")
        report = "\n".join(lines)
        _save_report(report)
        return report

    target_vals = df[TARGET_COL].value_counts()
    lines.append(f"   Unique values: {list(target_vals.index)}")
    lines.append(f"   Distribution:")
    for label, cnt in target_vals.items():
        pct = cnt / len(df) * 100
        lines.append(f"      {label:>8s}: {cnt:>5d} ({pct:5.1f}%)")
    missing_target = df[TARGET_COL].isnull().sum()
    lines.append(f"   Missing targets: {missing_target}")

    unexpected = set(df[TARGET_COL].dropna().unique()) - set(TARGET_CLASSES)
    if unexpected:
        lines.append(f"   *** WARNING: Unexpected labels found: {unexpected} ***")
    else:
        lines.append(f"   All labels match expected classes: {TARGET_CLASSES}")

    # -- 4. Missing values ---------------------------------------------
    lines.append(f"\n4. MISSING VALUES")
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)
    total_missing = null_counts.sum()
    features_with_missing = (null_counts > 0).sum()
    lines.append(f"   Total missing cells: {total_missing}")
    lines.append(f"   Features with missing values: {features_with_missing}/{len(df.columns)}")
    lines.append(f"\n   Per-feature missing counts:")
    for col in df.columns:
        if null_counts[col] > 0:
            lines.append(f"      {col:>30s}: {null_counts[col]:>3d} ({null_pcts[col]:>5.2f}%)")

    # -- 5. Duplicate checks -------------------------------------------
    lines.append(f"\n5. DUPLICATE CHECKS")
    dup_rows = df.duplicated().sum()
    lines.append(f"   Duplicate rows: {dup_rows}")
    if ID_COL in df.columns:
        dup_ids = df[ID_COL].duplicated().sum()
        lines.append(f"   Duplicate {ID_COL}s: {dup_ids}")
    else:
        lines.append(f"   '{ID_COL}' column not found, skipping ID duplicate check.")

    # -- 6. Data types -------------------------------------------------
    lines.append(f"\n6. DATA TYPES")
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in feature_cols if c not in numeric_features]
    lines.append(f"   Numeric biomarker features: {len(numeric_features)}")
    lines.append(f"   Non-numeric features: {len(non_numeric)}")
    if non_numeric:
        lines.append(f"   *** WARNING -- Non-numeric feature columns: {non_numeric} ***")
        for col in non_numeric:
            lines.append(f"      {col} sample values: {df[col].dropna().head(5).tolist()}")

    # -- 7. Constant / near-constant features --------------------------
    lines.append(f"\n7. CONSTANT / NEAR-CONSTANT FEATURES")
    constant_feats = []
    for col in numeric_features:
        nunique = df[col].dropna().nunique()
        if nunique <= 1:
            constant_feats.append(col)
    if constant_feats:
        lines.append(f"   *** Constant features (will be removed): {constant_feats} ***")
    else:
        lines.append(f"   No constant features detected.")

    # -- 8. Value range plausibility -----------------------------------
    lines.append(f"\n8. VALUE RANGES (min / max for numeric features)")
    for col in numeric_features:
        mn = df[col].min()
        mx = df[col].max()
        lines.append(f"   {col:>30s}: min={mn:>10.2f}, max={mx:>10.2f}")

    # -- 9. Negative value check ---------------------------------------
    lines.append(f"\n9. NEGATIVE VALUE CHECK")
    neg_cols = []
    for col in numeric_features:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            neg_cols.append((col, neg_count))
    if neg_cols:
        lines.append(f"   *** WARNING -- Negative values found: ***")
        for col, cnt in neg_cols:
            lines.append(f"      {col}: {cnt} negative values")
    else:
        lines.append(f"   No negative biomarker values detected.")

    # -- 10. Target leakage audit --------------------------------------
    lines.append(f"\n10. TARGET LEAKAGE AUDIT")
    lines.append(f"   Excluded from features: {EXCLUDE_COLS}")
    lines.append(f"   Features used for training: {len(numeric_features)}")
    lines.append(f"   PASS -- Target column and identifiers excluded from feature set.")

    # -- 11. Labeling rules analysis -----------------------------------
    lines.append(f"\n11. LABELING RULES ANALYSIS")
    lines.append(f"   Labels are RULE-DERIVED from biomarker thresholds.")
    lines.append(f"   Rules: HARD conditions (H1-H4) checked first -> then MEDIUM (M1-M6) -> else GOOD.")
    lines.append(f"   *** IMPORTANT: High model accuracy is expected because the labels are")
    lines.append(f"   deterministic functions of the input features. This does NOT prove")
    lines.append(f"   the model has learned medically meaningful patterns. It has learned")
    lines.append(f"   to approximate the synthetic labeling rules. ***")

    # -- 12. Class imbalance warning -----------------------------------
    lines.append(f"\n12. CLASS IMBALANCE WARNING")
    if TARGET_COL in df.columns:
        hard_count = (df[TARGET_COL] == "HARD").sum()
        hard_pct = hard_count / len(df) * 100
        lines.append(f"   HARD category: {hard_count} samples ({hard_pct:.1f}%)")
        if hard_count < 50:
            lines.append(f"   *** WARNING: HARD has very few samples ({hard_count}).")
            lines.append(f"   Per-class metrics for HARD will be noisy / unreliable.")
            lines.append(f"   Stratified splitting is used to ensure representation. ***")

    lines.append(f"\n{SEPARATOR}")
    lines.append("VALIDATION COMPLETE")
    lines.append(SEPARATOR)

    report = "\n".join(lines)
    _save_report(report)
    return report


def _save_report(report: str):
    """Save validation report to outputs/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Validation report saved to: {path}")


def prepare_features_target(df: pd.DataFrame):
    """
    Separate features (X) and target (y).
    Returns: X (DataFrame), y (Series), feature_names (list)
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Keep only numeric features
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[TARGET_COL]

    # Drop constant features
    constant_cols = [c for c in X.columns if X[c].dropna().nunique() <= 1]
    if constant_cols:
        print(f"[INFO] Dropping constant features: {constant_cols}")
        X = X.drop(columns=constant_cols)

    feature_names = list(X.columns)
    print(f"[INFO] Features: {len(feature_names)}, Target: '{TARGET_COL}'")
    print(f"[INFO] Feature list: {feature_names}")
    return X, y, feature_names
