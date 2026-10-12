#!/usr/bin/env python3
"""
NutriCare ML -- Dual-Mode Assessment & Prediction Engine
=========================================================
Supports two distinct, clinically responsible workflows:

Mode A: General Health Classification
    Uses the trained supervised Random Forest model (39 biomarkers)
    to predict synthetic demonstration tiers (GOOD / MEDIUM / HARD).

Mode B: Condition-Specific Pattern Assessment
    A modular, transparent rule-based clinical pattern engine that prompts
    ONLY for biomarkers relevant to a selected condition (1 of 10 panels),
    validates units and physiological boundaries, detects abnormal patterns,
    explains clinical significance, and provides preventive guidance.

DISCLAIMER: This system is strictly an educational prototype built on synthetic
            data. It is NOT a medical device, diagnosis tool, or clinical decision
            support system. Always consult licensed healthcare professionals.
"""

import sys
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Ensure UTF-8 output encoding where supported on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_PATH, METADATA_PATH, SEPARATOR
from src.biomarker_validation import BiomarkerValidator, BIOMARKER_SPECS
from src.disease_patterns import DiseasePatternEngine, CONDITION_PANELS
from src.disease_explainer import DiseaseExplainer

LABEL_MAPPING = {
    "GOOD": "Good",
    "MEDIUM": "Okay",
    "HARD": "Poor",
}


# -- Synthetic demo samples for Mode A ---------------------------------
DEMO_SAMPLES_MODE_A = {
    "GOOD_example": {
        "Glucose_mg_dL": 88.0, "HbA1c_pct": 5.1, "Hemoglobin_g_dL": 14.5,
        "Total_Cholesterol_mg_dL": 175.0, "LDL_Cholesterol_mg_dL": 95.0,
        "HDL_Cholesterol_mg_dL": 58.0, "Triglycerides_mg_dL": 110.0,
        "Vitamin_D_ng_mL": 35.0, "Vitamin_B12_pg_mL": 450.0,
        "Ferritin_ng_mL": 80.0, "Creatinine_mg_dL": 0.85, "Urea_mg_dL": 28.0,
        "ALT_U_L": 22.0, "AST_U_L": 20.0, "WBC_10^9_L": 6.5,
        "Platelet_10^9_L": 250.0, "RBC_10^12_L": 4.8, "TSH_mIU_L": 2.0,
        "Albumin_g_dL": 4.2, "CRP_mg_L": 1.0, "eGFR_mL_min_1_73m2": 105.0,
        "Bilirubin_mg_dL": 0.6, "ALP_U_L": 65.0, "Sodium_mmol_L": 140.0,
        "Potassium_mmol_L": 4.2, "Calcium_mg_dL": 9.5, "Magnesium_mg_dL": 2.0,
        "Folate_ng_mL": 12.0, "MCV_fL": 88.0, "MCH_pg": 29.0,
        "MCHC_g_dL": 34.0, "RDW_pct": 12.5, "Neutrophil_10^9_L": 3.5,
        "Lymphocyte_10^9_L": 2.0, "ApoB_mg_dL": 80.0,
        "Lipoprotein_a_mg_dL": 15.0, "Homocysteine_umol_L": 8.0,
        "GGT_U_L": 25.0, "UACR_mg_g": 10.0,
    },
    "MEDIUM_example": {
        "Glucose_mg_dL": 135.0, "HbA1c_pct": 6.8, "Hemoglobin_g_dL": 13.5,
        "Total_Cholesterol_mg_dL": 220.0, "LDL_Cholesterol_mg_dL": 145.0,
        "HDL_Cholesterol_mg_dL": 42.0, "Triglycerides_mg_dL": 180.0,
        "Vitamin_D_ng_mL": 18.0, "Vitamin_B12_pg_mL": 350.0,
        "Ferritin_ng_mL": 60.0, "Creatinine_mg_dL": 1.1, "Urea_mg_dL": 35.0,
        "ALT_U_L": 45.0, "AST_U_L": 38.0, "WBC_10^9_L": 7.5,
        "Platelet_10^9_L": 280.0, "RBC_10^12_L": 4.5, "TSH_mIU_L": 2.5,
        "Albumin_g_dL": 3.8, "CRP_mg_L": 6.0, "eGFR_mL_min_1_73m2": 85.0,
        "Bilirubin_mg_dL": 0.9, "ALP_U_L": 75.0, "Sodium_mmol_L": 138.0,
        "Potassium_mmol_L": 4.0, "Calcium_mg_dL": 9.2, "Magnesium_mg_dL": 1.9,
        "Folate_ng_mL": 10.0, "MCV_fL": 90.0, "MCH_pg": 28.5,
        "MCHC_g_dL": 33.5, "RDW_pct": 13.5, "Neutrophil_10^9_L": 4.5,
        "Lymphocyte_10^9_L": 2.2, "ApoB_mg_dL": 110.0,
        "Lipoprotein_a_mg_dL": 30.0, "Homocysteine_umol_L": 11.0,
        "GGT_U_L": 40.0, "UACR_mg_g": 35.0,
    },
    "HARD_example": {
        "Glucose_mg_dL": 195.0, "HbA1c_pct": 9.0, "Hemoglobin_g_dL": 9.5,
        "Total_Cholesterol_mg_dL": 280.0, "LDL_Cholesterol_mg_dL": 195.0,
        "HDL_Cholesterol_mg_dL": 35.0, "Triglycerides_mg_dL": 300.0,
        "Vitamin_D_ng_mL": 12.0, "Vitamin_B12_pg_mL": 180.0,
        "Ferritin_ng_mL": 10.0, "Creatinine_mg_dL": 2.5, "Urea_mg_dL": 55.0,
        "ALT_U_L": 130.0, "AST_U_L": 125.0, "WBC_10^9_L": 13.0,
        "Platelet_10^9_L": 140.0, "RBC_10^12_L": 3.8, "TSH_mIU_L": 0.05,
        "Albumin_g_dL": 3.0, "CRP_mg_L": 25.0, "eGFR_mL_min_1_73m2": 35.0,
        "Bilirubin_mg_dL": 3.5, "ALP_U_L": 120.0, "Sodium_mmol_L": 128.0,
        "Potassium_mmol_L": 6.5, "Calcium_mg_dL": 8.0, "Magnesium_mg_dL": 1.4,
        "Folate_ng_mL": 4.0, "MCV_fL": 78.0, "MCH_pg": 25.0,
        "MCHC_g_dL": 31.0, "RDW_pct": 17.0, "Neutrophil_10^9_L": 8.0,
        "Lymphocyte_10^9_L": 1.0, "ApoB_mg_dL": 160.0,
        "Lipoprotein_a_mg_dL": 80.0, "Homocysteine_umol_L": 20.0,
        "GGT_U_L": 100.0, "UACR_mg_g": 350.0,
    },
}

# -- Synthetic demo samples for Mode B ---------------------------------
DEMO_SAMPLES_MODE_B = {
    "diabetes": {
        "Glucose_mg_dL": 185.0,
        "HbA1c_pct": 8.2,
    },
    "anemia": {
        "Hemoglobin_g_dL": 9.4,
        "MCV_fL": 74.0,
        "RDW_pct": 17.2,
        "Ferritin_ng_mL": 12.0,
    },
    "dyslipidemia": {
        "Total_Cholesterol_mg_dL": 265.0,
        "LDL_Cholesterol_mg_dL": 188.0,
        "HDL_Cholesterol_mg_dL": 36.0,
        "Triglycerides_mg_dL": 245.0,
        "ApoB_mg_dL": 140.0,
    },
    "kidney": {
        "Creatinine_mg_dL": 2.1,
        "eGFR_mL_min_1_73m2": 34.0,
        "Urea_mg_dL": 68.0,
        "UACR_mg_g": 340.0,
    },
}


def load_model():
    """Load the saved pipeline and metadata for Mode A."""
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("Please run 'python train.py' first.")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)
    metadata = {}
    if os.path.isfile(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return pipeline, metadata


# ===========================================================================
# MODE A: General Health Classification (Supervised ML)
# ===========================================================================

def predict_mode_a(pipeline, metadata, values: Dict[str, Any], show_proba: bool = True):
    """Predicts GOOD / MEDIUM / HARD using the 39-biomarker ML pipeline."""
    feature_names = metadata.get("input_features", [])
    if not feature_names:
        print("[ERROR] No feature list in metadata. Retrain the model.")
        return

    row = {}
    warnings = []
    for feat in feature_names:
        val = values.get(feat)
        if val is not None and str(val).strip() != "":
            try:
                row[feat] = float(val)
            except (ValueError, TypeError):
                warnings.append(f"[!] '{feat}': non-numeric value '{val}' -> treated as missing")
                row[feat] = np.nan
        else:
            row[feat] = np.nan
            warnings.append(f"[i] '{feat}': missing -> will be imputed by median")

    if warnings and len(warnings) < len(feature_names):
        print(f"\n   Input warnings:")
        for w in warnings[:5]:
            print(f"      {w}")
        if len(warnings) > 5:
            print(f"      ... and {len(warnings) - 5} more missing features imputed.")

    X_input = pd.DataFrame([row], columns=feature_names)
    prediction = pipeline.predict(X_input)[0]
    display_label = LABEL_MAPPING.get(prediction, prediction)

    print(f"\n   {'='*55}")
    print(f"   >>> MODE A PREDICTED CATEGORY: {display_label} (internal: {prediction})")
    print(f"   {'='*55}")

    if show_proba:
        probas = pipeline.predict_proba(X_input)[0]
        classes = pipeline.classes_
        print(f"\n   Model Class Probabilities (Synthetic Training Distribution):")
        for cls, prob in zip(classes, probas):
            mapped_cls = LABEL_MAPPING.get(cls, cls)
            bar = "#" * int(prob * 30)
            print(f"      {mapped_cls:>8s} ({cls}): {prob:.4f}  {bar}")

    print(f"\n   [!] MODE A NOTICE: This category reflects synthetic dataset tiers")
    print(f"   (Good = normal bounds, Okay = moderate elevation, Poor = critical derangement).")
    print(f"   These represent learned patterns from synthetic data and are NOT clinical diagnoses.")
    return display_label


def interactive_mode_a(pipeline, metadata):
    """Interactive input for all 39 biomarkers."""
    feature_names = metadata.get("input_features", [])
    print(f"\n{SEPARATOR}")
    print("  MODE A: GENERAL HEALTH CLASSIFICATION (39 BIOMARKERS)")
    print(f"{SEPARATOR}")
    print("   Enter biomarker values below. Press Enter to skip (imputed by pipeline).")
    print("   Type 'q' to quit.\n")

    values = {}
    for i, feat in enumerate(feature_names, 1):
        spec = BIOMARKER_SPECS.get(feat, {})
        d_name = spec.get("display_name", feat)
        unit = spec.get("canonical_unit", "")
        try:
            prompt = f"   [{i:>2}/{len(feature_names)}] {d_name} ({unit}): "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Input cancelled.")
            return

        if user_input.lower() == "q":
            print("[INFO] Quit.")
            return

        values[feat] = None if user_input == "" else user_input

    predict_mode_a(pipeline, metadata, values)


# ===========================================================================
# MODE B: Condition-Specific Assessment (Targeted Clinical Panels)
# ===========================================================================

def display_condition_report(assessment, condition_id: str):
    """Displays a rich, structured, and clinically responsible report."""
    print(f"\n{SEPARATOR}")
    print(f"  CONDITION ASSESSMENT REPORT: {assessment.condition_name.upper()}")
    print(f"{SEPARATOR}")

    # 1. Overall Status
    status_tag = f"[{assessment.status}]"
    print(f"\n  Pattern Status: {status_tag}")

    # 2. Entered Biomarkers & Validation
    print(f"\n  [1] Biomarkers Evaluated:")
    print(f"  {'-'*55}")
    for b_name, val in assessment.biomarkers_evaluated.items():
        spec = BIOMARKER_SPECS.get(b_name, {})
        d_name = spec.get("display_name", b_name)
        unit = spec.get("canonical_unit", "")
        ref_low = spec.get("ref_low", "?")
        ref_high = spec.get("ref_high", "?")
        if val is not None:
            print(f"    * {d_name:<34} : {val:>8.2f} {unit:<8} (Ref: {ref_low}-{ref_high})")
        else:
            print(f"    * {d_name:<34} : [NOT PROVIDED]")

    # 3. Patterns & Findings
    print(f"\n  [2] Identified Patterns & Clinical Findings:")
    print(f"  {'-'*55}")
    if assessment.patterns_detected:
        for p in assessment.patterns_detected:
            print(f"    * Pattern: {p}")
    if assessment.findings_detail:
        for f in assessment.findings_detail:
            print(f"      - {f}")

    # 4. Clinical Caveats & Missing Context
    print(f"\n  [3] Differential Factors & Missing Clinical Tests:")
    print(f"  {'-'*55}")
    if assessment.clinical_caveats:
        for c in assessment.clinical_caveats:
            print(f"    [!] {c}")
    if assessment.unmeasured_clinical_context:
        print(f"    [Tests Absent from Synthetic Dataset]:")
        for u in assessment.unmeasured_clinical_context:
            print(f"      - {u}")

    # 5. Educational Explanations & Potential Complications
    exp = DiseaseExplainer.get_explanation(condition_id)
    print(f"\n  [4] Educational Overview & Physiological Role:")
    print(f"  {'-'*55}")
    print(f"    {exp['description']}\n")
    print(f"    Why This Pattern Warrants Attention:")
    print(f"    {exp['why_warrants_attention']}\n")
    print(f"    Potential Complications If Unmanaged:")
    for comp in exp.get("potential_complications", []):
        print(f"      * {comp}")

    # 6. General Guidance & Prevention
    print(f"\n  [5] General Preventive & Maintenance Guidance:")
    print(f"  {'-'*55}")
    for g in exp.get("general_guidance", []):
        print(f"    + {g}")

    # 7. Healthcare Consultation Advice
    print(f"\n  [6] Healthcare Consultation Guidance:")
    print(f"  {'-'*55}")
    print(f"    {exp['consultation_advice']}")

    # 8. Prominent Medical Disclaimer
    print(f"\n  {'='*55}")
    print(f"  MEDICAL DISCLAIMER:")
    print(f"  This assessment is an educational demonstration based on synthetic data")
    print(f"  and rule-based pattern matching. It is NOT a medical diagnosis.")
    print(f"  Do not alter any medical treatment or take supplements based on this output.")
    print(f"  Always seek guidance from a qualified, licensed physician.")
    print(f"  {'='*55}\n")


def interactive_mode_b():
    """Interactive targeted workflow: asks only for biomarkers of selected condition."""
    while True:
        print(f"\n{SEPARATOR}")
        print("  MODE B: SELECT A TARGETED CONDITION PANEL")
        print(f"{SEPARATOR}")
        condition_list = list(CONDITION_PANELS.keys())
        for idx, c_id in enumerate(condition_list, 1):
            c_name = CONDITION_PANELS[c_id]["name"]
            print(f"   [{idx:>2}] {c_name}")
        print("   [ 0] Return to Main Menu\n")

        try:
            choice = input("  Select condition (0-10): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Returning to main menu.")
            return

        if choice == "0":
            return

        try:
            c_idx = int(choice) - 1
            if not (0 <= c_idx < len(condition_list)):
                print("[!] Invalid selection. Please enter a number between 0 and 10.")
                continue
        except ValueError:
            print("[!] Invalid selection. Enter a number.")
            continue

        selected_id = condition_list[c_idx]
        panel = CONDITION_PANELS[selected_id]

        print(f"\n{SEPARATOR}")
        print(f"  ASSESSMENT: {panel['name'].upper()}")
        print(f"{SEPARATOR}")
        print("  Enter values for the relevant biomarkers below.")
        print("  Fields marked with [*] are essential for this assessment.")
        print("  Press Enter to skip optional fields. Type 'q' to cancel.\n")

        all_panel_fields = panel["required"] + panel["optional"]
        user_inputs: Dict[str, Any] = {}

        for feat in all_panel_fields:
            spec = BIOMARKER_SPECS.get(feat, {})
            d_name = spec.get("display_name", feat)
            unit = spec.get("canonical_unit", "")
            is_req = feat in panel["required"]
            req_mark = "[*]" if is_req else "[optional]"

            while True:
                try:
                    val_str = input(f"   {d_name} ({unit}) {req_mark}: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[INFO] Assessment cancelled.")
                    return

                if val_str.lower() == "q":
                    print("[INFO] Returning to condition menu.")
                    break

                if val_str == "":
                    if is_req:
                        print(f"      [!] '{d_name}' is essential for this assessment.")
                        confirm = input("      Proceed without it? Assessment may be incomplete (y/n): ").strip().lower()
                        if confirm == "y":
                            user_inputs[feat] = None
                            break
                        else:
                            continue
                    else:
                        user_inputs[feat] = None
                        break

                # Validate single input in real-time
                v_res = BiomarkerValidator.validate_single(feat, val_str)
                if not v_res.is_valid:
                    for issue in v_res.issues:
                        print(f"      [ERROR] {issue.message}")
                    retry = input("      Re-enter value? (y/n): ").strip().lower()
                    if retry != "n":
                        continue
                    else:
                        user_inputs[feat] = val_str
                        break
                else:
                    for issue in v_res.issues:
                        if issue.severity == "UNIT_MISMATCH":
                            print(f"      [WARNING] {issue.message}")
                        elif issue.severity == "CRITICAL":
                            print(f"      [CRITICAL ALERT] {issue.message}")
                        elif issue.severity == "WARNING":
                            print(f"      [Notice] {issue.message}")
                    user_inputs[feat] = v_res.cleaned_value
                    break

            if val_str.lower() == "q":
                break

        if val_str.lower() == "q":
            continue

        # Run assessment engine
        assessment = DiseasePatternEngine.evaluate_condition(selected_id, user_inputs)
        display_condition_report(assessment, selected_id)


def run_demo_mode_a(pipeline, metadata):
    """Run synthetic demo cases for Mode A."""
    print(f"\n{SEPARATOR}")
    print("  DEMONSTRATION: MODE A (GENERAL HEALTH CLASSIFICATION)")
    print(f"{SEPARATOR}")
    for name, values in DEMO_SAMPLES_MODE_A.items():
        print(f"\n{'-'*55}")
        print(f"  Demo Sample: {name}")
        print(f"{'-'*55}")
        predict_mode_a(pipeline, metadata, values)


def run_demo_mode_b():
    """Run synthetic demo cases for Mode B."""
    print(f"\n{SEPARATOR}")
    print("  DEMONSTRATION: MODE B (CONDITION-SPECIFIC PATTERNS)")
    print(f"{SEPARATOR}")
    for cond_id, values in DEMO_SAMPLES_MODE_B.items():
        assessment = DiseasePatternEngine.evaluate_condition(cond_id, values)
        display_condition_report(assessment, cond_id)


# ===========================================================================
# MAIN MENU DISPATCHER
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NutriCare ML -- Blood Biomarker Assessment System",
        epilog="DISCLAIMER: Educational demonstration only. Not for clinical diagnosis.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run Mode A (General 3-Class ML Model) demonstrations",
    )
    parser.add_argument(
        "--demo-conditions", action="store_true",
        help="Run Mode B (Condition-Specific Pattern) demonstrations",
    )
    parser.add_argument(
        "--mode", choices=["general", "condition"],
        help="Directly launch into 'general' (Mode A) or 'condition' (Mode B)",
    )
    args = parser.parse_args()

    pipeline, metadata = load_model()

    if args.demo:
        run_demo_mode_a(pipeline, metadata)
        return

    if args.demo_conditions:
        run_demo_mode_b()
        return

    if args.mode == "general":
        interactive_mode_a(pipeline, metadata)
        return
    elif args.mode == "condition":
        interactive_mode_b()
        return

    # Interactive top-level menu
    while True:
        print(f"\n{SEPARATOR}")
        print("  NUTRICARE CLINICAL BIOMARKER ASSESSMENT SYSTEM")
        print(f"{SEPARATOR}")
        print("   [1] Mode A: General Health Classification (39-Biomarker ML Model)")
        print("   [2] Mode B: Condition-Specific Assessment (Targeted Clinical Panel)")
        print("   [3] Run Automated Demonstrations (Both Mode A and Mode B)")
        print("   [4] Exit\n")

        try:
            choice = input("  Select an option (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting NutriCare.")
            break

        if choice == "1":
            interactive_mode_a(pipeline, metadata)
        elif choice == "2":
            interactive_mode_b()
        elif choice == "3":
            run_demo_mode_a(pipeline, metadata)
            run_demo_mode_b()
        elif choice == "4" or choice.lower() == "q":
            print("[INFO] Exiting NutriCare. Thank you.")
            break
        else:
            print("[!] Invalid selection. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
