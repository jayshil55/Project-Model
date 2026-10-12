"""
NutriCare ML -- Biomarker Input Validation Engine
Author: Senior ML & Health-Data Software Engineer
Date: 2026-10-12

Provides unit-aware, clinical-bound validation for all 39 biomarkers.
Handles:
  * Non-numeric and malformed input
  * Missing and empty values
  * Biologically impossible values
  * Suspicious unit mismatches (e.g. mmol/L vs mg/dL, g/L vs g/dL)
  * Critical / panic lab values requiring urgent clinical attention
  * Inconsistent / extreme values requiring user verification

CRITICAL PRINCIPLES:
  * Never silently clip, drop, or alter input values.
  * Clearly separate invalid data from extreme but possible results.
  * All validation rules are unit-aware and medically referenced.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class ValidationIssue:
    severity: str  # 'ERROR', 'CRITICAL', 'WARNING', 'UNIT_MISMATCH', 'INFO'
    biomarker: str
    message: str
    suggested_unit: Optional[str] = None
    converted_estimate: Optional[float] = None


@dataclass
class BiomarkerValidationResult:
    biomarker: str
    raw_value: Any
    cleaned_value: Optional[float]
    is_valid: bool
    is_missing: bool
    is_critical: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "ERROR" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity in ("WARNING", "UNIT_MISMATCH", "CRITICAL") for i in self.issues)


# ---------------------------------------------------------------------------
# Biomarker Specification Table
# Defines canonical units, physiological bounds, reference ranges,
# critical limits, and unit-mismatch detection thresholds.
# ---------------------------------------------------------------------------

BIOMARKER_SPECS: Dict[str, Dict[str, Any]] = {
    "Glucose_mg_dL": {
        "display_name": "Serum Glucose",
        "canonical_unit": "mg/dL",
        "min_physiologic": 10.0,
        "max_physiologic": 2500.0,
        "ref_low": 70.0,
        "ref_high": 140.0,
        "critical_low": 50.0,
        "critical_high": 400.0,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/L",
            "detected_range": (2.0, 30.0),
            "conversion_factor": 18.0182,  # mg/dL = mmol/L * 18.0182
            "hint": "Value is in 2.0-30.0 range. Did you enter glucose in mmol/L instead of mg/dL?",
        },
    },
    "HbA1c_pct": {
        "display_name": "Glycated Hemoglobin (HbA1c)",
        "canonical_unit": "%",
        "min_physiologic": 3.0,
        "max_physiologic": 25.0,
        "ref_low": 4.0,
        "ref_high": 5.6,
        "critical_low": None,
        "critical_high": 14.0,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/mol (IFCC)",
            "detected_range": (20.0, 150.0),
            "conversion_formula": lambda v: (0.09148 * v) + 2.152,  # NGSP % from IFCC mmol/mol
            "hint": "Value is in 20-150 range. Did you enter HbA1c in IFCC mmol/mol instead of %?",
        },
    },
    "Hemoglobin_g_dL": {
        "display_name": "Hemoglobin",
        "canonical_unit": "g/dL",
        "min_physiologic": 2.0,
        "max_physiologic": 26.0,
        "ref_low": 12.0,
        "ref_high": 17.5,
        "critical_low": 7.0,
        "critical_high": 20.0,
        "unit_mismatch_detection": {
            "alt_unit": "g/L",
            "detected_range": (50.0, 220.0),
            "conversion_factor": 0.1,  # g/dL = g/L / 10
            "hint": "Value is in 50-220 range. Did you enter hemoglobin in g/L instead of g/dL?",
        },
    },
    "Total_Cholesterol_mg_dL": {
        "display_name": "Total Cholesterol",
        "canonical_unit": "mg/dL",
        "min_physiologic": 30.0,
        "max_physiologic": 1200.0,
        "ref_low": 120.0,
        "ref_high": 200.0,
        "critical_low": None,
        "critical_high": 400.0,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/L",
            "detected_range": (2.0, 15.0),
            "conversion_factor": 38.67,  # mg/dL = mmol/L * 38.67
            "hint": "Value is in 2.0-15.0 range. Did you enter total cholesterol in mmol/L instead of mg/dL?",
        },
    },
    "LDL_Cholesterol_mg_dL": {
        "display_name": "LDL Cholesterol",
        "canonical_unit": "mg/dL",
        "min_physiologic": 10.0,
        "max_physiologic": 1000.0,
        "ref_low": 50.0,
        "ref_high": 100.0,
        "critical_low": None,
        "critical_high": 250.0,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/L",
            "detected_range": (1.0, 12.0),
            "conversion_factor": 38.67,
            "hint": "Value is in 1.0-12.0 range. Did you enter LDL in mmol/L instead of mg/dL?",
        },
    },
    "HDL_Cholesterol_mg_dL": {
        "display_name": "HDL Cholesterol",
        "canonical_unit": "mg/dL",
        "min_physiologic": 5.0,
        "max_physiologic": 200.0,
        "ref_low": 40.0,
        "ref_high": 80.0,
        "critical_low": 20.0,
        "critical_high": None,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/L",
            "detected_range": (0.5, 4.0),
            "conversion_factor": 38.67,
            "hint": "Value is in 0.5-4.0 range. Did you enter HDL in mmol/L instead of mg/dL?",
        },
    },
    "Triglycerides_mg_dL": {
        "display_name": "Triglycerides",
        "canonical_unit": "mg/dL",
        "min_physiologic": 10.0,
        "max_physiologic": 5000.0,
        "ref_low": 40.0,
        "ref_high": 150.0,
        "critical_low": None,
        "critical_high": 500.0,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/L",
            "detected_range": (0.5, 10.0),
            "conversion_factor": 88.57,  # mg/dL = mmol/L * 88.57
            "hint": "Value is in 0.5-10.0 range. Did you enter triglycerides in mmol/L instead of mg/dL?",
        },
    },
    "Vitamin_D_ng_mL": {
        "display_name": "25-Hydroxy Vitamin D",
        "canonical_unit": "ng/mL",
        "min_physiologic": 1.0,
        "max_physiologic": 300.0,
        "ref_low": 30.0,
        "ref_high": 100.0,
        "critical_low": 10.0,
        "critical_high": 150.0,
        "unit_mismatch_detection": {
            "alt_unit": "nmol/L",
            "detected_range": (75.0, 250.0),
            "conversion_factor": 0.4006,  # ng/mL = nmol/L / 2.496
            "hint": "Value is in 75-250 range. Did you enter vitamin D in nmol/L instead of ng/mL?",
        },
    },
    "Vitamin_B12_pg_mL": {
        "display_name": "Vitamin B12",
        "canonical_unit": "pg/mL",
        "min_physiologic": 20.0,
        "max_physiologic": 4000.0,
        "ref_low": 200.0,
        "ref_high": 950.0,
        "critical_low": 150.0,
        "critical_high": None,
        "unit_mismatch_detection": {
            "alt_unit": "pmol/L",
            "detected_range": (80.0, 1500.0),
            "conversion_factor": 1.355,  # pg/mL = pmol/L * 1.355 approx
            "hint": None,
        },
    },
    "Ferritin_ng_mL": {
        "display_name": "Serum Ferritin",
        "canonical_unit": "ng/mL",
        "min_physiologic": 1.0,
        "max_physiologic": 20000.0,
        "ref_low": 30.0,
        "ref_high": 300.0,
        "critical_low": 10.0,
        "critical_high": 1000.0,
    },
    "Creatinine_mg_dL": {
        "display_name": "Serum Creatinine",
        "canonical_unit": "mg/dL",
        "min_physiologic": 0.1,
        "max_physiologic": 30.0,
        "ref_low": 0.6,
        "ref_high": 1.3,
        "critical_low": 0.3,
        "critical_high": 4.0,
        "unit_mismatch_detection": {
            "alt_unit": "umol/L",
            "detected_range": (40.0, 1500.0),
            "conversion_factor": 0.011312,  # mg/dL = umol/L / 88.4
            "hint": "Value is in 40-1500 range. Did you enter creatinine in umol/L instead of mg/dL?",
        },
    },
    "Urea_mg_dL": {
        "display_name": "Serum Urea (Direct Urea)",
        "canonical_unit": "mg/dL",
        "min_physiologic": 2.0,
        "max_physiologic": 400.0,
        "ref_low": 15.0,
        "ref_high": 45.0,
        "critical_low": 5.0,
        "critical_high": 120.0,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/L",
            "detected_range": (1.5, 30.0),
            "conversion_factor": 6.006,  # mg/dL = mmol/L * 6.006
            "hint": "Value is in 1.5-30.0 range. Did you enter urea in mmol/L instead of mg/dL?",
        },
    },
    "ALT_U_L": {
        "display_name": "Alanine Aminotransferase (ALT)",
        "canonical_unit": "U/L",
        "min_physiologic": 1.0,
        "max_physiologic": 10000.0,
        "ref_low": 7.0,
        "ref_high": 45.0,
        "critical_low": None,
        "critical_high": 300.0,
    },
    "AST_U_L": {
        "display_name": "Aspartate Aminotransferase (AST)",
        "canonical_unit": "U/L",
        "min_physiologic": 1.0,
        "max_physiologic": 10000.0,
        "ref_low": 8.0,
        "ref_high": 40.0,
        "critical_low": None,
        "critical_high": 300.0,
    },
    "WBC_10^9_L": {
        "display_name": "White Blood Cell Count (WBC)",
        "canonical_unit": "10^9/L",
        "min_physiologic": 0.1,
        "max_physiologic": 300.0,
        "ref_low": 4.0,
        "ref_high": 11.0,
        "critical_low": 2.0,
        "critical_high": 30.0,
        "unit_mismatch_detection": {
            "alt_unit": "/uL or /mm^3",
            "detected_range": (1000.0, 50000.0),
            "conversion_factor": 0.001,
            "hint": "Value is in 1,000-50,000 range. Did you enter WBC in /uL instead of 10^9/L?",
        },
    },
    "Platelet_10^9_L": {
        "display_name": "Platelet Count",
        "canonical_unit": "10^9/L",
        "min_physiologic": 2.0,
        "max_physiologic": 2000.0,
        "ref_low": 150.0,
        "ref_high": 450.0,
        "critical_low": 30.0,
        "critical_high": 800.0,
        "unit_mismatch_detection": {
            "alt_unit": "/uL",
            "detected_range": (50000.0, 1000000.0),
            "conversion_factor": 0.001,
            "hint": "Value is in 50,000-1,000,000 range. Did you enter platelets in /uL instead of 10^9/L?",
        },
    },
    "RBC_10^12_L": {
        "display_name": "Red Blood Cell Count (RBC)",
        "canonical_unit": "10^12/L",
        "min_physiologic": 0.5,
        "max_physiologic": 10.0,
        "ref_low": 4.0,
        "ref_high": 5.9,
        "critical_low": 2.0,
        "critical_high": 7.5,
        "unit_mismatch_detection": {
            "alt_unit": "million/uL",
            "detected_range": None,  # 10^12/L equals million/uL directly
        },
    },
    "TSH_mIU_L": {
        "display_name": "Thyroid-Stimulating Hormone (TSH)",
        "canonical_unit": "mIU/L",
        "min_physiologic": 0.01,
        "max_physiologic": 200.0,
        "ref_low": 0.40,
        "ref_high": 4.00,
        "critical_low": 0.05,
        "critical_high": 15.0,
    },
    "Albumin_g_dL": {
        "display_name": "Serum Albumin",
        "canonical_unit": "g/dL",
        "min_physiologic": 0.5,
        "max_physiologic": 8.0,
        "ref_low": 3.5,
        "ref_high": 5.2,
        "critical_low": 2.0,
        "critical_high": 6.0,
        "unit_mismatch_detection": {
            "alt_unit": "g/L",
            "detected_range": (15.0, 60.0),
            "conversion_factor": 0.1,
            "hint": "Value is in 15-60 range. Did you enter albumin in g/L instead of g/dL?",
        },
    },
    "CRP_mg_L": {
        "display_name": "C-Reactive Protein (CRP)",
        "canonical_unit": "mg/L",
        "min_physiologic": 0.01,
        "max_physiologic": 600.0,
        "ref_low": 0.0,
        "ref_high": 5.0,
        "critical_low": None,
        "critical_high": 50.0,
        "unit_mismatch_detection": {
            "alt_unit": "mg/dL",
            "detected_range": (0.001, 0.49),
            "conversion_factor": 10.0,
            "hint": None,
        },
    },
    "eGFR_mL_min_1_73m2": {
        "display_name": "Estimated Glomerular Filtration Rate (eGFR)",
        "canonical_unit": "mL/min/1.73m^2",
        "min_physiologic": 1.0,
        "max_physiologic": 200.0,
        "ref_low": 90.0,
        "ref_high": 130.0,
        "critical_low": 15.0,
        "critical_high": None,
    },
    "Bilirubin_mg_dL": {
        "display_name": "Total Bilirubin",
        "canonical_unit": "mg/dL",
        "min_physiologic": 0.05,
        "max_physiologic": 50.0,
        "ref_low": 0.2,
        "ref_high": 1.2,
        "critical_low": None,
        "critical_high": 12.0,
        "unit_mismatch_detection": {
            "alt_unit": "umol/L",
            "detected_range": (5.0, 200.0),
            "conversion_factor": 0.05847,  # mg/dL = umol/L / 17.1
            "hint": "Value is in 5-200 range. Did you enter bilirubin in umol/L instead of mg/dL?",
        },
    },
    "ALP_U_L": {
        "display_name": "Alkaline Phosphatase (ALP)",
        "canonical_unit": "U/L",
        "min_physiologic": 5.0,
        "max_physiologic": 2000.0,
        "ref_low": 40.0,
        "ref_high": 130.0,
        "critical_low": 20.0,
        "critical_high": 400.0,
    },
    "Sodium_mmol_L": {
        "display_name": "Serum Sodium",
        "canonical_unit": "mmol/L",
        "min_physiologic": 80.0,
        "max_physiologic": 200.0,
        "ref_low": 135.0,
        "ref_high": 145.0,
        "critical_low": 120.0,
        "critical_high": 155.0,
    },
    "Potassium_mmol_L": {
        "display_name": "Serum Potassium",
        "canonical_unit": "mmol/L",
        "min_physiologic": 1.0,
        "max_physiologic": 12.0,
        "ref_low": 3.5,
        "ref_high": 5.0,
        "critical_low": 2.8,
        "critical_high": 6.2,
    },
    "Calcium_mg_dL": {
        "display_name": "Serum Calcium",
        "canonical_unit": "mg/dL",
        "min_physiologic": 2.0,
        "max_physiologic": 25.0,
        "ref_low": 8.5,
        "ref_high": 10.5,
        "critical_low": 6.5,
        "critical_high": 13.0,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/L",
            "detected_range": (1.5, 3.5),
            "conversion_factor": 4.008,  # mg/dL = mmol/L * 4.008
            "hint": "Value is in 1.5-3.5 range. Did you enter calcium in mmol/L instead of mg/dL?",
        },
    },
    "Magnesium_mg_dL": {
        "display_name": "Serum Magnesium",
        "canonical_unit": "mg/dL",
        "min_physiologic": 0.2,
        "max_physiologic": 15.0,
        "ref_low": 1.7,
        "ref_high": 2.4,
        "critical_low": 1.0,
        "critical_high": 4.5,
        "unit_mismatch_detection": {
            "alt_unit": "mmol/L",
            "detected_range": (0.6, 1.3),
            "conversion_factor": 2.43,
            "hint": "Value is in 0.6-1.3 range. Did you enter magnesium in mmol/L instead of mg/dL?",
        },
    },
    "Folate_ng_mL": {
        "display_name": "Serum Folate",
        "canonical_unit": "ng/mL",
        "min_physiologic": 0.5,
        "max_physiologic": 100.0,
        "ref_low": 4.0,
        "ref_high": 20.0,
        "critical_low": 2.0,
        "critical_high": None,
        "unit_mismatch_detection": {
            "alt_unit": "nmol/L",
            "detected_range": (10.0, 60.0),
            "conversion_factor": 0.441,  # ng/mL = nmol/L * 0.441
            "hint": None,
        },
    },
    "MCV_fL": {
        "display_name": "Mean Corpuscular Volume (MCV)",
        "canonical_unit": "fL",
        "min_physiologic": 30.0,
        "max_physiologic": 160.0,
        "ref_low": 80.0,
        "ref_high": 100.0,
        "critical_low": 65.0,
        "critical_high": 125.0,
    },
    "MCH_pg": {
        "display_name": "Mean Corpuscular Hemoglobin (MCH)",
        "canonical_unit": "pg",
        "min_physiologic": 10.0,
        "max_physiologic": 55.0,
        "ref_low": 27.0,
        "ref_high": 33.0,
        "critical_low": 20.0,
        "critical_high": 40.0,
    },
    "MCHC_g_dL": {
        "display_name": "MCHC",
        "canonical_unit": "g/dL",
        "min_physiologic": 15.0,
        "max_physiologic": 45.0,
        "ref_low": 32.0,
        "ref_high": 36.0,
        "critical_low": 26.0,
        "critical_high": 38.0,
        "unit_mismatch_detection": {
            "alt_unit": "g/L",
            "detected_range": (280.0, 380.0),
            "conversion_factor": 0.1,
            "hint": "Value is in 280-380 range. Did you enter MCHC in g/L instead of g/dL?",
        },
    },
    "RDW_pct": {
        "display_name": "Red Cell Distribution Width (RDW)",
        "canonical_unit": "%",
        "min_physiologic": 8.0,
        "max_physiologic": 40.0,
        "ref_low": 11.5,
        "ref_high": 15.0,
        "critical_low": None,
        "critical_high": 25.0,
    },
    "Neutrophil_10^9_L": {
        "display_name": "Absolute Neutrophil Count",
        "canonical_unit": "10^9/L",
        "min_physiologic": 0.05,
        "max_physiologic": 100.0,
        "ref_low": 2.0,
        "ref_high": 7.5,
        "critical_low": 0.5,  # Agranulocytosis
        "critical_high": 20.0,
    },
    "Lymphocyte_10^9_L": {
        "display_name": "Absolute Lymphocyte Count",
        "canonical_unit": "10^9/L",
        "min_physiologic": 0.05,
        "max_physiologic": 50.0,
        "ref_low": 1.0,
        "ref_high": 4.0,
        "critical_low": 0.5,
        "critical_high": 10.0,
    },
    "ApoB_mg_dL": {
        "display_name": "Apolipoprotein B (ApoB)",
        "canonical_unit": "mg/dL",
        "min_physiologic": 10.0,
        "max_physiologic": 400.0,
        "ref_low": 50.0,
        "ref_high": 90.0,
        "critical_low": None,
        "critical_high": 150.0,
        "unit_mismatch_detection": {
            "alt_unit": "g/L",
            "detected_range": (0.4, 2.5),
            "conversion_factor": 100.0,
            "hint": "Value is in 0.4-2.5 range. Did you enter ApoB in g/L instead of mg/dL?",
        },
    },
    "Lipoprotein_a_mg_dL": {
        "display_name": "Lipoprotein(a)",
        "canonical_unit": "mg/dL",
        "min_physiologic": 0.1,
        "max_physiologic": 500.0,
        "ref_low": 0.0,
        "ref_high": 30.0,
        "critical_low": None,
        "critical_high": 100.0,
        "unit_mismatch_detection": {
            "alt_unit": "nmol/L",
            "detected_range": (75.0, 300.0),
            "conversion_factor": 0.4,  # Approximate mg/dL ~ nmol/L * 0.4
            "hint": None,
        },
    },
    "Homocysteine_umol_L": {
        "display_name": "Serum Homocysteine",
        "canonical_unit": "umol/L",
        "min_physiologic": 1.0,
        "max_physiologic": 250.0,
        "ref_low": 5.0,
        "ref_high": 15.0,
        "critical_low": None,
        "critical_high": 50.0,
    },
    "GGT_U_L": {
        "display_name": "Gamma-Glutamyl Transferase (GGT)",
        "canonical_unit": "U/L",
        "min_physiologic": 2.0,
        "max_physiologic": 2000.0,
        "ref_low": 9.0,
        "ref_high": 50.0,
        "critical_low": None,
        "critical_high": 250.0,
    },
    "UACR_mg_g": {
        "display_name": "Urine Albumin-to-Creatinine Ratio (UACR)",
        "canonical_unit": "mg/g",
        "min_physiologic": 0.1,
        "max_physiologic": 8000.0,
        "ref_low": 0.0,
        "ref_high": 30.0,
        "critical_low": None,
        "critical_high": 1000.0,
        "unit_mismatch_detection": {
            "alt_unit": "mg/mmol",
            "detected_range": (3.0, 100.0),
            "conversion_factor": 8.84,  # mg/g = mg/mmol * 8.84
            "hint": "Value is in 3-100 range. Did you enter UACR in mg/mmol instead of mg/g?",
        },
    },
}


class BiomarkerValidator:
    """
    Robust clinical input validator for individual biomarkers and panels.
    Ensures safe, unit-aware parsing without silent imputation or clipping.
    """

    @staticmethod
    def validate_single(biomarker: str, value: Any) -> BiomarkerValidationResult:
        """
        Validates an individual biomarker measurement against clinical bounds and unit specs.
        """
        # Unknown biomarker check
        if biomarker not in BIOMARKER_SPECS:
            return BiomarkerValidationResult(
                biomarker=biomarker,
                raw_value=value,
                cleaned_value=None,
                is_valid=False,
                is_missing=False,
                is_critical=False,
                issues=[
                    ValidationIssue(
                        severity="ERROR",
                        biomarker=biomarker,
                        message=f"Unknown biomarker '{biomarker}'. Not part of the 39-feature schema.",
                    )
                ],
            )

        spec = BIOMARKER_SPECS[biomarker]
        issues: List[ValidationIssue] = []

        # 1. Missing / None / Empty check
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return BiomarkerValidationResult(
                biomarker=biomarker,
                raw_value=value,
                cleaned_value=None,
                is_valid=True,
                is_missing=True,
                is_critical=False,
                issues=[
                    ValidationIssue(
                        severity="INFO",
                        biomarker=biomarker,
                        message=f"{spec['display_name']} was omitted (missing).",
                    )
                ],
            )

        # 2. Non-numeric parsing check
        num_val: Optional[float] = None
        try:
            if isinstance(value, (int, float)):
                num_val = float(value)
            elif isinstance(value, str):
                num_val = float(value.strip())
            else:
                raise ValueError("Unsupported type")
        except (ValueError, TypeError):
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    biomarker=biomarker,
                    message=f"Non-numeric input '{value}' for {spec['display_name']} ({spec['canonical_unit']}). Number expected.",
                )
            )
            return BiomarkerValidationResult(
                biomarker=biomarker,
                raw_value=value,
                cleaned_value=None,
                is_valid=False,
                is_missing=False,
                is_critical=False,
                issues=issues,
            )

        # Check for NaN / Infinity
        import math
        if math.isnan(num_val) or math.isinf(num_val):
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    biomarker=biomarker,
                    message=f"Invalid float value '{num_val}' for {spec['display_name']}.",
                )
            )
            return BiomarkerValidationResult(
                biomarker=biomarker,
                raw_value=value,
                cleaned_value=None,
                is_valid=False,
                is_missing=False,
                is_critical=False,
                issues=issues,
            )

        # 3. Negative value check
        if num_val < 0:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    biomarker=biomarker,
                    message=f"Negative value {num_val} is physiologically impossible for {spec['display_name']}.",
                )
            )
            return BiomarkerValidationResult(
                biomarker=biomarker,
                raw_value=value,
                cleaned_value=num_val,
                is_valid=False,
                is_missing=False,
                is_critical=False,
                issues=issues,
            )

        # 4. Biologically impossible limits check (outside min/max physiologic)
        min_p = spec.get("min_physiologic")
        max_p = spec.get("max_physiologic")

        if min_p is not None and num_val < min_p:
            # Check if this might be a unit mismatch before rejecting
            mismatch_info = spec.get("unit_mismatch_detection")
            if mismatch_info and mismatch_info.get("detected_range"):
                r_low, r_high = mismatch_info["detected_range"]
                if r_low <= num_val <= r_high:
                    hint = mismatch_info.get("hint") or f"Value matches range for {mismatch_info['alt_unit']}."
                    issues.append(
                        ValidationIssue(
                            severity="UNIT_MISMATCH",
                            biomarker=biomarker,
                            message=f"SUSPECTED UNIT MISMATCH: {hint}",
                            suggested_unit=mismatch_info["alt_unit"],
                        )
                    )

            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    biomarker=biomarker,
                    message=f"Value {num_val} {spec['canonical_unit']} is below biological survival limit ({min_p} {spec['canonical_unit']}).",
                )
            )
            return BiomarkerValidationResult(
                biomarker=biomarker,
                raw_value=value,
                cleaned_value=num_val,
                is_valid=False,
                is_missing=False,
                is_critical=False,
                issues=issues,
            )

        if max_p is not None and num_val > max_p:
            # Check if this could be an alternate unit (e.g. Platelets in /uL instead of 10^9/L)
            mismatch_info = spec.get("unit_mismatch_detection")
            if mismatch_info and mismatch_info.get("detected_range"):
                r_low, r_high = mismatch_info["detected_range"]
                if r_low <= num_val <= r_high:
                    hint = mismatch_info.get("hint") or f"Value matches range for {mismatch_info['alt_unit']}."
                    issues.append(
                        ValidationIssue(
                            severity="UNIT_MISMATCH",
                            biomarker=biomarker,
                            message=f"SUSPECTED UNIT MISMATCH: {hint}",
                            suggested_unit=mismatch_info["alt_unit"],
                        )
                    )

            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    biomarker=biomarker,
                    message=f"Value {num_val} {spec['canonical_unit']} exceeds maximum biological survival limit ({max_p} {spec['canonical_unit']}).",
                )
            )
            return BiomarkerValidationResult(
                biomarker=biomarker,
                raw_value=value,
                cleaned_value=num_val,
                is_valid=False,
                is_missing=False,
                is_critical=False,
                issues=issues,
            )

        # 5. Suspicious unit mismatch heuristic (within plausible range, but fits another unit)
        mismatch_info = spec.get("unit_mismatch_detection")
        if mismatch_info and mismatch_info.get("detected_range"):
            r_low, r_high = mismatch_info["detected_range"]
            # If the value falls squarely in the alternate unit's expected range
            # and is notably discordant with normal canonical reference ranges
            if r_low <= num_val <= r_high:
                # E.g. Glucose in mmol/L: 3.5 to 15.0 mmol/L is normal/diabetic in mmol/L,
                # but in mg/dL is extreme hypoglycemia (< 50 mg/dL)!
                if num_val < (spec.get("ref_low", 0) * 0.5):
                    hint = mismatch_info.get("hint") or f"Did you enter {mismatch_info['alt_unit']}?"
                    issues.append(
                        ValidationIssue(
                            severity="UNIT_MISMATCH",
                            biomarker=biomarker,
                            message=f"SUSPECTED UNIT MISMATCH: {hint}",
                            suggested_unit=mismatch_info["alt_unit"],
                        )
                    )

        # 6. Critical / Panic values check
        is_critical = False
        crit_low = spec.get("critical_low")
        crit_high = spec.get("critical_high")

        if crit_low is not None and num_val < crit_low:
            is_critical = True
            issues.append(
                ValidationIssue(
                    severity="CRITICAL",
                    biomarker=biomarker,
                    message=f"CRITICAL LAB VALUE: {num_val} {spec['canonical_unit']} is below critical threshold ({crit_low} {spec['canonical_unit']}). Requires immediate medical attention.",
                )
            )

        if crit_high is not None and num_val > crit_high:
            is_critical = True
            issues.append(
                ValidationIssue(
                    severity="CRITICAL",
                    biomarker=biomarker,
                    message=f"CRITICAL LAB VALUE: {num_val} {spec['canonical_unit']} exceeds critical threshold ({crit_high} {spec['canonical_unit']}). Requires immediate medical attention.",
                )
            )

        # 7. Unusual / Extreme Warning check (outside typical healthy reference range)
        ref_l = spec.get("ref_low")
        ref_h = spec.get("ref_high")
        if not is_critical:
            if ref_l is not None and num_val < ref_l:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        biomarker=biomarker,
                        message=f"{spec['display_name']} ({num_val} {spec['canonical_unit']}) is below standard healthy reference range ({ref_l} - {ref_h} {spec['canonical_unit']}).",
                    )
                )
            elif ref_h is not None and num_val > ref_h:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        biomarker=biomarker,
                        message=f"{spec['display_name']} ({num_val} {spec['canonical_unit']}) is above standard healthy reference range ({ref_l} - {ref_h} {spec['canonical_unit']}).",
                    )
                )

        return BiomarkerValidationResult(
            biomarker=biomarker,
            raw_value=value,
            cleaned_value=num_val,
            is_valid=True,
            is_missing=False,
            is_critical=is_critical,
            issues=issues,
        )

    @classmethod
    def validate_panel(
        cls, inputs: Dict[str, Any], required_biomarkers: List[str]
    ) -> Tuple[Dict[str, BiomarkerValidationResult], List[str]]:
        """
        Validates an entire dictionary of biomarker measurements for a condition panel.
        Returns:
            - Dict mapping biomarker_name to BiomarkerValidationResult
            - List of global panel errors / missing required fields
        """
        results: Dict[str, BiomarkerValidationResult] = {}
        panel_errors: List[str] = []

        for b_name, val in inputs.items():
            results[b_name] = cls.validate_single(b_name, val)

        # Verify whether required biomarkers are missing
        for req in required_biomarkers:
            if req not in results or results[req].is_missing:
                spec = BIOMARKER_SPECS.get(req, {})
                d_name = spec.get("display_name", req)
                unit = spec.get("canonical_unit", "")
                panel_errors.append(
                    f"Required biomarker missing: {d_name} ({req}) [{unit}]. Assessment cannot proceed without this key indicator."
                )
            elif not results[req].is_valid:
                spec = BIOMARKER_SPECS.get(req, {})
                d_name = spec.get("display_name", req)
                panel_errors.append(
                    f"Required biomarker invalid: {d_name} ({req}) contains errors that prevent reliable assessment."
                )

        return results, panel_errors
