"""
NutriCare TUI -- Chat-First ML Engine & Natural Language Biomarker Parser
Author: Senior Python TUI Engineer & Health-Data Software Engineer
Date: 2026-10-12

Provides:
  - Local natural language parsing for 39 blood biomarkers
  - Alias & synonym resolution (e.g. "blood sugar" -> Glucose_mg_dL, "a1c" -> HbA1c_pct)
  - Extraction of values with optional units from conversational statements
  - Integration with BiomarkerValidator for clinical boundaries & unit mismatches
  - Conversational execution of the Random Forest ML pipeline
  - Rich chat bubble rendering for Hermes Agent / Gemini style TUI
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.config import MODEL_PATH, METADATA_PATH
from src.biomarker_validation import BiomarkerValidator, BIOMARKER_SPECS
from src.tui.charts import render_horizontal_bar, render_biomarker_range_indicator
from src.tui.widgets import record_session_assessment

# ---------------------------------------------------------------------------
# Comprehensive Biomarker Alias Dictionary (Maps spoken/typed names to keys)
# ---------------------------------------------------------------------------
BIOMARKER_ALIASES: Dict[str, str] = {
    # Glycemic
    "glucose": "Glucose_mg_dL",
    "blood sugar": "Glucose_mg_dL",
    "fasting glucose": "Glucose_mg_dL",
    "fbg": "Glucose_mg_dL",
    "fbs": "Glucose_mg_dL",
    "sugar": "Glucose_mg_dL",
    "hba1c": "HbA1c_pct",
    "a1c": "HbA1c_pct",
    "glycated hemoglobin": "HbA1c_pct",
    "glycohemoglobin": "HbA1c_pct",

    # Hematology
    "hemoglobin": "Hemoglobin_g_dL",
    "hgb": "Hemoglobin_g_dL",
    "hb": "Hemoglobin_g_dL",
    "mcv": "MCV_fL",
    "mean corpuscular volume": "MCV_fL",
    "mch": "MCH_pg",
    "mchc": "MCHC_g_dL",
    "rdw": "RDW_pct",
    "wbc": "WBC_10^9_L",
    "white blood cells": "WBC_10^9_L",
    "white cells": "WBC_10^9_L",
    "leukocytes": "WBC_10^9_L",
    "platelets": "Platelet_10^9_L",
    "platelet": "Platelet_10^9_L",
    "plt": "Platelet_10^9_L",
    "thrombocytes": "Platelet_10^9_L",
    "rbc": "RBC_10^12_L",
    "red blood cells": "RBC_10^12_L",
    "erythrocytes": "RBC_10^12_L",
    "neutrophils": "Neutrophil_10^9_L",
    "neutrophil": "Neutrophil_10^9_L",
    "neuts": "Neutrophil_10^9_L",
    "lymphocytes": "Lymphocyte_10^9_L",
    "lymphocyte": "Lymphocyte_10^9_L",
    "lymphs": "Lymphocyte_10^9_L",

    # Lipids
    "total cholesterol": "Total_Cholesterol_mg_dL",
    "cholesterol": "Total_Cholesterol_mg_dL",
    "tc": "Total_Cholesterol_mg_dL",
    "ldl": "LDL_Cholesterol_mg_dL",
    "ldl-c": "LDL_Cholesterol_mg_dL",
    "ldl cholesterol": "LDL_Cholesterol_mg_dL",
    "bad cholesterol": "LDL_Cholesterol_mg_dL",
    "hdl": "HDL_Cholesterol_mg_dL",
    "hdl-c": "HDL_Cholesterol_mg_dL",
    "hdl cholesterol": "HDL_Cholesterol_mg_dL",
    "good cholesterol": "HDL_Cholesterol_mg_dL",
    "triglycerides": "Triglycerides_mg_dL",
    "triglyceride": "Triglycerides_mg_dL",
    "tg": "Triglycerides_mg_dL",
    "trigs": "Triglycerides_mg_dL",
    "apob": "ApoB_mg_dL",
    "apolipoprotein b": "ApoB_mg_dL",
    "lipoprotein a": "Lipoprotein_a_mg_dL",
    "lp(a)": "Lipoprotein_a_mg_dL",
    "lpa": "Lipoprotein_a_mg_dL",

    # Vitamins & Iron
    "vitamin d": "Vitamin_D_ng_mL",
    "vit d": "Vitamin_D_ng_mL",
    "25(oh)d": "Vitamin_D_ng_mL",
    "25-oh d": "Vitamin_D_ng_mL",
    "calcidiol": "Vitamin_D_ng_mL",
    "vitamin b12": "Vitamin_B12_pg_mL",
    "b12": "Vitamin_B12_pg_mL",
    "vit b12": "Vitamin_B12_pg_mL",
    "cobalamin": "Vitamin_B12_pg_mL",
    "folate": "Folate_ng_mL",
    "folic acid": "Folate_ng_mL",
    "ferritin": "Ferritin_ng_mL",
    "serum ferritin": "Ferritin_ng_mL",
    "homocysteine": "Homocysteine_umol_L",
    "hcy": "Homocysteine_umol_L",

    # Renal & Metabolic
    "creatinine": "Creatinine_mg_dL",
    "creat": "Creatinine_mg_dL",
    "cr": "Creatinine_mg_dL",
    "serum creatinine": "Creatinine_mg_dL",
    "egfr": "eGFR_mL_min_1_73m2",
    "gfr": "eGFR_mL_min_1_73m2",
    "urea": "Urea_mg_dL",
    "bun": "Urea_mg_dL",
    "blood urea nitrogen": "Urea_mg_dL",
    "uacr": "UACR_mg_g",
    "urine albumin": "UACR_mg_g",
    "microalbumin": "UACR_mg_g",

    # Hepatic & Enzymes
    "alt": "ALT_U_L",
    "sgpt": "ALT_U_L",
    "alanine aminotransferase": "ALT_U_L",
    "ast": "AST_U_L",
    "sgot": "AST_U_L",
    "aspartate aminotransferase": "AST_U_L",
    "alp": "ALP_U_L",
    "alkaline phosphatase": "ALP_U_L",
    "bilirubin": "Bilirubin_mg_dL",
    "total bilirubin": "Bilirubin_mg_dL",
    "tbil": "Bilirubin_mg_dL",
    "albumin": "Albumin_g_dL",
    "ggt": "GGT_U_L",
    "gamma-glutamyl transferase": "GGT_U_L",

    # Thyroid & Inflammation
    "tsh": "TSH_mIU_L",
    "thyroid stimulating hormone": "TSH_mIU_L",
    "crp": "CRP_mg_L",
    "c-reactive protein": "CRP_mg_L",
    "hs-crp": "CRP_mg_L",

    # Electrolytes
    "sodium": "Sodium_mmol_L",
    "na": "Sodium_mmol_L",
    "potassium": "Potassium_mmol_L",
    "k": "Potassium_mmol_L",
    "calcium": "Calcium_mg_dL",
    "ca": "Calcium_mg_dL",
    "magnesium": "Magnesium_mg_dL",
    "mg": "Magnesium_mg_dL",
}

# Also register canonical feature names and display names from BIOMARKER_SPECS
for feat_name, spec in BIOMARKER_SPECS.items():
    BIOMARKER_ALIASES[feat_name.lower()] = feat_name
    d_name = spec.get("display_name", "").lower()
    if d_name and d_name not in BIOMARKER_ALIASES:
        BIOMARKER_ALIASES[d_name] = feat_name

# Sorted aliases by length descending so longer phrases match first (e.g. 'total cholesterol' before 'cholesterol')
SORTED_ALIASES = sorted(BIOMARKER_ALIASES.keys(), key=len, reverse=True)

LABEL_MAPPING = {
    "GOOD": "Good",
    "MEDIUM": "Okay",
    "HARD": "Poor",
}


class ChatMessage:
    """Represents a single message in the conversational history."""

    def __init__(self, role: str, text: str, rich_content: Any = None):
        self.role = role  # "assistant" or "user"
        self.text = text
        self.rich_content = rich_content


class ChatEngine:
    """
    Conversational AI assistant for biomarker collection and machine learning evaluation.
    Operates 100% locally with rule-based NLP extraction and local Scikit-Learn pipeline.
    """

    def __init__(self):
        self.collected_biomarkers: Dict[str, float] = {}
        self.history: List[ChatMessage] = []
        self._add_welcome_message()

    def _add_welcome_message(self):
        welcome_text = (
            "Welcome to NutriCare ML. I am your local biomarker analysis assistant.\n\n"
            "You can enter individual lab values or provide multiple biomarkers in natural language:\n"
            "  • 'Glucose is 95 mg/dL and HbA1c is 5.2%'\n"
            "  • 'Cholesterol: 190, HDL: 55, LDL: 110, Triglycerides: 125'\n"
            "  • 'Creatinine 0.9, eGFR 95, Vitamin D 35'\n\n"
            "Available actions: 'run prediction', 'summary', 'clear', 'preset good/okay/poor', or 'help'."
        )
        self.history.append(ChatMessage("assistant", welcome_text))

    def reset(self):
        """Resets conversation history and collected biomarkers."""
        self.collected_biomarkers.clear()
        self.history.clear()
        self._add_welcome_message()

    def clear_values(self) -> str:
        """Clears collected biomarker values without wiping full conversation."""
        count = len(self.collected_biomarkers)
        self.collected_biomarkers.clear()
        return f"Cleared {count} stored biomarker values. You can enter new measurements or load a preset."

    def set_biomarkers(self, values: Dict[str, Any]):
        """Sets or updates biomarker values from dictionary."""
        for feat, val in values.items():
            if val is not None and str(val).strip() != "":
                try:
                    self.collected_biomarkers[feat] = float(val)
                except ValueError:
                    pass

    def extract_biomarkers(self, text: str) -> Tuple[Dict[str, float], List[str], List[str]]:
        """
        Extracts biomarkers and numeric values from a user text statement.
        Returns:
            - extracted_valid: Dict[feature_name, float_val]
            - notices: List of info/warning strings (out of range, unit mismatch)
            - errors: List of error strings (unparseable, biologically impossible)
        """
        extracted: Dict[str, float] = {}
        notices: List[str] = []
        errors: List[str] = []

        lower_text = text.lower()

        # Find occurrences of aliases
        for alias in SORTED_ALIASES:
            # Pattern: alias followed by optional punctuation/words, then number, then optional unit
            # e.g., "glucose is 95", "glucose: 95 mg/dl", "glucose = 95", "glucose 95"
            pattern = re.compile(
                rf"\b{re.escape(alias)}\b\s*(?:is|of|=|:|\bto\b)?\s*([+-]?[0-9]*\.?[0-9]+)\s*([a-zA-Z/%_^\-\d]+)?",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(lower_text))
            if matches:
                feat_key = BIOMARKER_ALIASES[alias]
                if feat_key not in extracted:
                    # Pick the first match
                    m = matches[0]
                    raw_num_str = m.group(1)
                    raw_unit_str = m.group(2) if m.group(2) else ""

                    # Validate with BiomarkerValidator
                    v_res = BiomarkerValidator.validate_single(feat_key, raw_num_str)
                    spec = BIOMARKER_SPECS.get(feat_key, {})
                    d_name = spec.get("display_name", feat_key)
                    c_unit = spec.get("canonical_unit", "")

                    if not v_res.is_valid:
                        err_msg = v_res.issues[0].message if v_res.issues else f"Invalid value: {raw_num_str}"
                        errors.append(f"• {d_name}: {err_msg}")
                    else:
                        cleaned_val = v_res.cleaned_value
                        extracted[feat_key] = cleaned_val

                        # Check for critical or warning issues
                        for issue in v_res.issues:
                            if issue.severity == "CRITICAL":
                                notices.append(f"⚠ CRITICAL VALUE: {d_name} ({cleaned_val} {c_unit}) requires urgent attention.")
                            elif issue.severity == "UNIT_MISMATCH":
                                notices.append(f"Notice: {issue.message}")
                            elif issue.severity == "WARNING":
                                notices.append(f"• {d_name} ({cleaned_val} {c_unit}) is outside standard reference range.")

        return extracted, notices, errors

    def handle_message(self, user_text: str) -> ChatMessage:
        """Processes user input message, executes commands or extraction, and appends to history."""
        cleaned = user_text.strip()
        self.history.append(ChatMessage("user", cleaned))

        lower = cleaned.lower()

        # 1. Help command
        if lower in ("help", "?", "commands", "options"):
            help_text = (
                "NutriCare ML Assistant Commands:\n\n"
                "  • Biomarker Entry: Type values naturally, e.g.:\n"
                "    - 'Glucose 95 mg/dL, HbA1c 5.2%'\n"
                "    - 'Creatinine 1.0, Urea 30, eGFR 90'\n"
                "  • 'run' / 'predict' : Execute Random Forest prediction on collected profile\n"
                "  • 'summary' / 'status' : View table of current biomarkers and reference ranges\n"
                "  • 'preset good' : Load benchmark healthy preset\n"
                "  • 'preset okay' : Load benchmark borderline preset\n"
                "  • 'preset poor' : Load benchmark high-risk preset\n"
                "  • 'clear' : Reset currently entered biomarker values\n"
                "  • 'manual' : Switch to full 39-biomarker manual entry form"
            )
            resp = ChatMessage("assistant", help_text)
            self.history.append(resp)
            return resp

        # 2. Clear command
        if lower in ("clear", "reset", "clear inputs", "clear values"):
            msg = self.clear_values()
            resp = ChatMessage("assistant", msg)
            self.history.append(resp)
            return resp

        # 3. Presets
        if any(kw in lower for kw in ("preset good", "good preset", "load good", "preset: good")):
            from src.tui.screens import PRESET_SAMPLES
            self.set_biomarkers(PRESET_SAMPLES["Good"])
            resp = ChatMessage("assistant", f"Loaded 'Good' benchmark preset ({len(PRESET_SAMPLES['Good'])} biomarkers). Say 'run' to analyze.")
            self.history.append(resp)
            return resp

        if any(kw in lower for kw in ("preset okay", "okay preset", "load okay", "preset medium", "preset: okay")):
            from src.tui.screens import PRESET_SAMPLES
            self.set_biomarkers(PRESET_SAMPLES["Okay"])
            resp = ChatMessage("assistant", f"Loaded 'Okay' benchmark preset ({len(PRESET_SAMPLES['Okay'])} biomarkers). Say 'run' to analyze.")
            self.history.append(resp)
            return resp

        if any(kw in lower for kw in ("preset poor", "poor preset", "load poor", "preset hard", "preset: poor")):
            from src.tui.screens import PRESET_SAMPLES
            self.set_biomarkers(PRESET_SAMPLES["Poor"])
            resp = ChatMessage("assistant", f"Loaded 'Poor' benchmark preset ({len(PRESET_SAMPLES['Poor'])} biomarkers). Say 'run' to analyze.")
            self.history.append(resp)
            return resp

        # 4. Summary command
        if lower in ("summary", "status", "list", "show", "values", "show summary"):
            if not self.collected_biomarkers:
                resp = ChatMessage("assistant", "No biomarkers entered yet. Enter values (e.g., 'Glucose 95 mg/dL') or load a preset ('preset good').")
                self.history.append(resp)
                return resp

            table = Table(title="Current Collected Biomarker Profile", box=box.ROUNDED, header_style="bold #7585E8")
            table.add_column("Biomarker", style="bold #E6E8F0")
            table.add_column("Value", justify="right", style="bold #43D9D0")
            table.add_column("Unit", style="#9299AA")
            table.add_column("Reference Range", style="#9299AA")
            table.add_column("Clinical Status", justify="center")

            for feat, val in sorted(self.collected_biomarkers.items()):
                spec = BIOMARKER_SPECS.get(feat, {})
                d_name = spec.get("display_name", feat)
                unit = spec.get("canonical_unit", "")
                r_low = spec.get("ref_low")
                r_high = spec.get("ref_high")
                c_low = spec.get("critical_low")
                c_high = spec.get("critical_high")

                status_tag = "[#4ECA78]NORMAL ✓[/]"
                if (c_low is not None and val < c_low) or (c_high is not None and val > c_high):
                    status_tag = "[bold #FFFFFF on #EF4444]CRITICAL ⚠[/]"
                elif r_low is not None and val < r_low:
                    status_tag = "[#EAB308]LOW ▼[/]"
                elif r_high is not None and val > r_high:
                    status_tag = "[#EF4444]HIGH ▲[/]"

                ref_str = f"{r_low} - {r_high}" if r_low is not None and r_high is not None else "-"
                table.add_row(d_name, f"{val:.2f}", unit, ref_str, status_tag)

            summary_text = f"Collected {len(self.collected_biomarkers)} of 39 biomarkers. Say 'run' to execute prediction."
            resp = ChatMessage("assistant", summary_text, rich_content=table)
            self.history.append(resp)
            return resp

        # 5. Prediction command
        if lower in ("run", "predict", "analyze", "run prediction", "evaluate", "assess"):
            if not self.collected_biomarkers:
                resp = ChatMessage("assistant", "Cannot run prediction: No biomarkers entered yet. Please enter values first or load a preset ('preset good').")
                self.history.append(resp)
                return resp

            pred_text, rich_card = self.run_ml_prediction()
            resp = ChatMessage("assistant", pred_text, rich_content=rich_card)
            self.history.append(resp)
            return resp

        # 6. Extract biomarkers from input text
        extracted, notices, errors = self.extract_biomarkers(cleaned)

        if not extracted and not errors:
            resp_text = (
                f"No biomarker values recognized in: '{cleaned}'.\n\n"
                "Try writing something like:\n"
                "  • 'Glucose is 95 mg/dL and HbA1c is 5.2%'\n"
                "  • 'Creatinine 1.1, eGFR 85'\n"
                "Or say 'help' for available commands."
            )
            resp = ChatMessage("assistant", resp_text)
            self.history.append(resp)
            return resp

        # Update stored biomarkers
        self.collected_biomarkers.update(extracted)

        # Build conversational response
        lines = []
        if extracted:
            lines.append(f"Recorded {len(extracted)} biomarker value{'s' if len(extracted) > 1 else ''}:")
            for feat, val in extracted.items():
                spec = BIOMARKER_SPECS.get(feat, {})
                d_name = spec.get("display_name", feat)
                unit = spec.get("canonical_unit", "")
                r_low = spec.get("ref_low")
                r_high = spec.get("ref_high")
                c_low = spec.get("critical_low")
                c_high = spec.get("critical_high")

                status_badge = "[NORMAL ✓]"
                if (c_low is not None and val < c_low) or (c_high is not None and val > c_high):
                    status_badge = "[CRITICAL ⚠]"
                elif r_low is not None and val < r_low:
                    status_badge = "[LOW ▼]"
                elif r_high is not None and val > r_high:
                    status_badge = "[HIGH ▲]"

                ref_info = f"(Ref: {r_low}-{r_high} {unit})" if r_low is not None and r_high is not None else f"({unit})"
                lines.append(f"  • {d_name}: {val} {unit} {status_badge} {ref_info}")

        if notices:
            lines.append("\nNotices:")
            for n in notices:
                lines.append(f"  {n}")

        if errors:
            lines.append("\nErrors / Unrecognized Values:")
            for e in errors:
                lines.append(f"  {e}")

        total_collected = len(self.collected_biomarkers)
        lines.append(f"\nProfile now has {total_collected} of 39 biomarkers.")
        lines.append("Would you like to add more biomarkers, view summary ('summary'), or run prediction ('run')?")

        resp = ChatMessage("assistant", "\n".join(lines))
        self.history.append(resp)
        return resp

    def run_ml_prediction(self) -> Tuple[str, Any]:
        """Runs the real trained Random Forest pipeline on collected biomarkers."""
        if not os.path.isfile(MODEL_PATH):
            return "ERROR: Trained model not found at models/nutricare_model.pkl. Run train.py first.", None

        pipeline = joblib.load(MODEL_PATH)
        metadata = {}
        if os.path.isfile(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        feature_names = metadata.get("input_features", list(BIOMARKER_SPECS.keys()))

        # Build feature row with NaNs for unmeasured features
        row = {}
        missing_count = 0
        for feat in feature_names:
            if feat in self.collected_biomarkers:
                row[feat] = self.collected_biomarkers[feat]
            else:
                row[feat] = np.nan
                missing_count += 1

        X_df = pd.DataFrame([row], columns=feature_names)
        raw_pred = pipeline.predict(X_df)[0]
        mapped_pred = LABEL_MAPPING.get(raw_pred, raw_pred)

        probas = pipeline.predict_proba(X_df)[0]
        classes = pipeline.classes_

        # Color
        pred_color = "#4ECA78" if mapped_pred == "Good" else ("#EAB308" if mapped_pred == "Okay" else "#EF4444")

        # Build Rich output text
        res_text = Text()
        res_text.append("═════════════════════════════════════════════════════════\n", style="#2E3345")
        res_text.append(f"  PREDICTED HEALTH STATUS: [{mapped_pred.upper()}]\n", style=f"bold {pred_color}")
        res_text.append("═════════════════════════════════════════════════════════\n", style="#2E3345")
        res_text.append(f"Model Classification: {mapped_pred} (internal label: {raw_pred})\n\n", style="#E6E8F0")

        res_text.append("Class Probability Distribution:\n", style="bold #43D9D0")
        for cls_name, prob in zip(classes, probas):
            mapped_cls = LABEL_MAPPING.get(cls_name, cls_name)
            bar = render_horizontal_bar(prob, 1.0, width=24, color="#5294E2")
            res_text.append(f"  {mapped_cls:<8}: {prob*100:>5.1f}%  ", style="bold #E6E8F0")
            res_text.append_text(bar)
            res_text.append("\n")

        res_text.append(f"\nMissing Features Imputed: {missing_count} of {len(feature_names)} features (via median)\n", style="#9299AA")

        # Range comparison for entered features
        entered_feats = [(f, v) for f, v in self.collected_biomarkers.items()]
        if entered_feats:
            res_text.append("\nMEASURED BIOMARKER REFERENCE RANGES:\n", style="bold #7585E8")
            res_text.append("─────────────────────────────────────────────────────────\n", style="#2E3345")
            for feat, val in entered_feats:
                spec = BIOMARKER_SPECS.get(feat, {})
                indicator = render_biomarker_range_indicator(
                    name=spec.get("display_name", feat)[:24],
                    val=val,
                    ref_low=spec.get("ref_low"),
                    ref_high=spec.get("ref_high"),
                    unit=spec.get("canonical_unit", ""),
                    critical_low=spec.get("critical_low"),
                    critical_high=spec.get("critical_high"),
                )
                res_text.append_text(indicator)

        res_text.append("\n[EDUCATIONAL NOTE] Probabilities reflect learned patterns from synthetic data.\n", style="italic #9299AA")
        res_text.append("This educational demonstration is NOT a clinical medical diagnosis.\n", style="italic #9299AA")

        # Record in session history
        record_session_assessment(
            mode="Chat Assistant",
            name="General ML Prediction",
            result_tag=mapped_pred,
            detail=f"{len(self.collected_biomarkers)} measured features, {missing_count} imputed",
        )

        overview_msg = f"Assessment complete: Predicted category is {mapped_pred} [{mapped_pred.upper()}]."
        return overview_msg, res_text
