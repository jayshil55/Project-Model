#!/usr/bin/env python3
"""
NutriCare ML -- Training Pipeline
=================================
Loads the Excel biomarker dataset, validates it, trains a Random Forest
classifier, evaluates on a held-out test set, and saves the model.

Usage:
    python train.py

Output files:
    models/nutricare_model.pkl       -- Fitted preprocessing + model pipeline
    models/model_metadata.json       -- Model metadata and evaluation metrics
    outputs/validation_report.txt    -- Dataset validation report
    outputs/evaluation_report.txt    -- Model evaluation report
    outputs/confusion_matrix.png     -- Confusion matrix visualization

DISCLAIMER: This is an educational ML demonstration using synthetic data.
            Model predictions are NOT clinical assessments.
"""
import sys
import os

# Ensure UTF-8 output encoding where supported on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SEPARATOR
from src.data_loader import load_dataset, validate_dataset, prepare_features_target, load_labeling_rules
from src.trainer import train_and_evaluate


def main():
    print(SEPARATOR)
    print("  NUTRICARE ML -- BLOOD BIOMARKER ANALYSIS")
    print("  Training Pipeline")
    print(SEPARATOR)

    # -- Step 1: Load dataset ------------------------------------------
    print("\n>> STEP 1: Loading dataset...")
    df = load_dataset()

    # -- Step 2: Load labeling rules (documentation only) -------------
    print("\n>> STEP 2: Loading labeling rules...")
    lr = load_labeling_rules()
    if not lr.empty:
        print("[INFO] Labeling rules loaded. Labels are RULE-DERIVED from biomarkers.")
        print("[INFO] High model accuracy is expected -- it reflects learned rules,")
        print("       not proof of medical prediction capability.")

    # -- Step 3: Validate dataset -------------------------------------
    print("\n>> STEP 3: Validating dataset...")
    report = validate_dataset(df)
    # Print report safely
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", errors="replace").decode("ascii"))

    # -- Step 4: Prepare features and target --------------------------
    print("\n>> STEP 4: Preparing features and target...")
    X, y, feature_names = prepare_features_target(df)

    # -- Step 5: Train and evaluate -----------------------------------
    print("\n>> STEP 5: Training and evaluating model...")
    metrics = train_and_evaluate(X, y, feature_names)

    # -- Final summary ------------------------------------------------
    print(f"\n{SEPARATOR}")
    print("[OK] TRAINING PIPELINE COMPLETE")
    print(SEPARATOR)
    print(f"   Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Test Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"   Baseline Acc:   {metrics['dummy_accuracy']:.4f}")
    print(f"   Baseline F1:    {metrics['dummy_macro_f1']:.4f}")
    print(f"\n   Files saved:")
    print(f"     * models/nutricare_model.pkl")
    print(f"     * models/model_metadata.json")
    print(f"     * outputs/validation_report.txt")
    print(f"     * outputs/evaluation_report.txt")
    print(f"     * outputs/confusion_matrix.png")
    print(f"\n   Run prediction demo:")
    print(f"     python predict.py")
    print(f"\n   [!] DISCLAIMER: Synthetic data -- not for clinical use.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
