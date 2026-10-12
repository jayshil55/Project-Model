"""
NutriCare ML -- Automated Test Suite for Biomarker Validation & Disease Patterns
Author: Senior ML & Health-Data Software Engineer
Date: 2026-10-12

Covers:
  * Normal synthetic inputs
  * Missing biomarkers & insufficient data
  * Non-numeric and malformed inputs
  * Biologically impossible and negative values
  * Suspicious unit mismatches (e.g. mmol/L vs mg/dL, g/L vs g/dL)
  * Critical / panic lab values
  * Multi-pattern recognition across all 10 condition panels
  * Edge-case condition handling
"""

import unittest
from src.biomarker_validation import BiomarkerValidator, BIOMARKER_SPECS
from src.disease_patterns import DiseasePatternEngine, CONDITION_PANELS
from src.disease_explainer import DiseaseExplainer


class TestBiomarkerValidation(unittest.TestCase):
    """Test suite for individual biomarker validation."""

    def test_normal_synthetic_inputs(self):
        """Verify standard in-range values pass validation with no errors."""
        res_gluc = BiomarkerValidator.validate_single("Glucose_mg_dL", 95.0)
        self.assertTrue(res_gluc.is_valid)
        self.assertEqual(res_gluc.cleaned_value, 95.0)
        self.assertFalse(res_gluc.has_errors)
        self.assertFalse(res_gluc.has_warnings)

        res_a1c = BiomarkerValidator.validate_single("HbA1c_pct", 5.2)
        self.assertTrue(res_a1c.is_valid)
        self.assertEqual(res_a1c.cleaned_value, 5.2)

        res_creat = BiomarkerValidator.validate_single("Creatinine_mg_dL", 0.9)
        self.assertTrue(res_creat.is_valid)

    def test_missing_and_empty_inputs(self):
        """Verify missing, None, and empty string inputs are correctly flagged as missing."""
        res_none = BiomarkerValidator.validate_single("Glucose_mg_dL", None)
        self.assertTrue(res_none.is_valid)
        self.assertTrue(res_none.is_missing)
        self.assertIsNone(res_none.cleaned_value)

        res_empty = BiomarkerValidator.validate_single("Hemoglobin_g_dL", "   ")
        self.assertTrue(res_empty.is_valid)
        self.assertTrue(res_empty.is_missing)

    def test_non_numeric_inputs(self):
        """Verify non-numeric string values trigger ERROR severity and are marked invalid."""
        res = BiomarkerValidator.validate_single("Glucose_mg_dL", "not_a_number")
        self.assertFalse(res.is_valid)
        self.assertTrue(res.has_errors)
        self.assertIn("Non-numeric", res.issues[0].message)

        res_malformed = BiomarkerValidator.validate_single("Creatinine_mg_dL", "12.3.4")
        self.assertFalse(res_malformed.is_valid)
        self.assertTrue(res_malformed.has_errors)

    def test_impossible_and_negative_values(self):
        """Verify negative and biologically impossible values are caught."""
        res_neg = BiomarkerValidator.validate_single("Hemoglobin_g_dL", -12.5)
        self.assertFalse(res_neg.is_valid)
        self.assertTrue(res_neg.has_errors)
        self.assertIn("Negative value", res_neg.issues[0].message)

        res_zero = BiomarkerValidator.validate_single("Glucose_mg_dL", 0.0)
        self.assertFalse(res_zero.is_valid)
        self.assertTrue(res_zero.has_errors)
        self.assertIn("survival limit", res_zero.issues[0].message)

        res_absurd = BiomarkerValidator.validate_single("HbA1c_pct", 85.0)
        self.assertFalse(res_absurd.is_valid)
        self.assertTrue(res_absurd.has_errors)

    def test_suspicious_unit_mismatches(self):
        """Verify common international unit mismatches trigger UNIT_MISMATCH warnings."""
        # Glucose: 5.5 mmol/L entered when expecting mg/dL
        res_gluc = BiomarkerValidator.validate_single("Glucose_mg_dL", 5.5)
        self.assertTrue(any(i.severity == "UNIT_MISMATCH" for i in res_gluc.issues))
        self.assertTrue(any("mmol/L" in i.message for i in res_gluc.issues))

        # Creatinine: 90 umol/L entered when expecting mg/dL
        res_creat = BiomarkerValidator.validate_single("Creatinine_mg_dL", 90.0)
        self.assertFalse(res_creat.is_valid)  # 90 mg/dL exceeds survival limit
        self.assertTrue(any(i.severity == "UNIT_MISMATCH" for i in res_creat.issues))
        self.assertTrue(any("umol/L" in i.message for i in res_creat.issues))

        # Hemoglobin: 140 g/L entered when expecting g/dL
        res_hb = BiomarkerValidator.validate_single("Hemoglobin_g_dL", 140.0)
        self.assertFalse(res_hb.is_valid)
        self.assertTrue(any(i.severity == "UNIT_MISMATCH" for i in res_hb.issues))
        self.assertTrue(any("g/L" in i.message for i in res_hb.issues))

        # Calcium: 2.2 mmol/L entered when expecting mg/dL
        res_ca = BiomarkerValidator.validate_single("Calcium_mg_dL", 2.2)
        self.assertTrue(any(i.severity == "UNIT_MISMATCH" for i in res_ca.issues))

    def test_critical_panic_values(self):
        """Verify life-threatening values trigger CRITICAL severity."""
        # Severe hyperkalemia (potassium = 6.8 mmol/L)
        res_k = BiomarkerValidator.validate_single("Potassium_mmol_L", 6.8)
        self.assertTrue(res_k.is_valid)
        self.assertTrue(res_k.is_critical)
        self.assertTrue(any(i.severity == "CRITICAL" for i in res_k.issues))

        # Severe hyponatremia (sodium = 118 mmol/L)
        res_na = BiomarkerValidator.validate_single("Sodium_mmol_L", 118.0)
        self.assertTrue(res_na.is_valid)
        self.assertTrue(res_na.is_critical)


class TestDiseasePatternEngine(unittest.TestCase):
    """Test suite for condition-specific pattern assessments."""

    def test_diabetes_normal_vs_abnormal(self):
        """Verify diabetes panel detects normal, prediabetic, and diabetic patterns."""
        normal_data = {"Glucose_mg_dL": 92.0, "HbA1c_pct": 5.2}
        res_norm = DiseasePatternEngine.evaluate_condition("diabetes", normal_data)
        self.assertEqual(res_norm.status, "NORMAL")
        self.assertIn("Normal Glycemic Indicators", res_norm.patterns_detected[0])

        diabetic_data = {"Glucose_mg_dL": 195.0, "HbA1c_pct": 8.8}
        res_diab = DiseasePatternEngine.evaluate_condition("diabetes", diabetic_data)
        self.assertEqual(res_diab.status, "ABNORMAL_PATTERN")
        self.assertIn("Elevated Glycemic Pattern", res_diab.patterns_detected[0])

        prediab_data = {"Glucose_mg_dL": 110.0, "HbA1c_pct": 5.9}
        res_pre = DiseasePatternEngine.evaluate_condition("diabetes", prediab_data)
        self.assertEqual(res_pre.status, "BORDERLINE")
        self.assertIn("Impaired Glycemic Regulation", res_pre.patterns_detected[0])

    def test_anemia_morphology_classification(self):
        """Verify microcytic vs macrocytic vs normocytic anemia detection."""
        # Microcytic
        micro_data = {"Hemoglobin_g_dL": 9.5, "MCV_fL": 72.0, "Ferritin_ng_mL": 11.0}
        res_micro = DiseasePatternEngine.evaluate_condition("anemia", micro_data)
        self.assertEqual(res_micro.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Microcytic" in p for p in res_micro.patterns_detected))

        # Macrocytic
        macro_data = {"Hemoglobin_g_dL": 10.2, "MCV_fL": 108.0, "Vitamin_B12_pg_mL": 140.0}
        res_macro = DiseasePatternEngine.evaluate_condition("anemia", macro_data)
        self.assertEqual(res_macro.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Macrocytic" in p for p in res_macro.patterns_detected))

    def test_iron_deficiency_pattern(self):
        """Verify absolute vs functional iron deficiency detection."""
        iron_def_data = {"Ferritin_ng_mL": 12.0, "Hemoglobin_g_dL": 10.5, "MCV_fL": 75.0, "RDW_pct": 16.5}
        res = DiseasePatternEngine.evaluate_condition("iron_deficiency", iron_def_data)
        self.assertEqual(res.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Iron Depletion" in p for p in res.patterns_detected))
        self.assertTrue(any("Iron Deficiency Anemia" in p for p in res.patterns_detected))

    def test_dyslipidemia_pattern(self):
        """Verify severe hypercholesterolemia and high ApoB detection."""
        lipid_data = {
            "Total_Cholesterol_mg_dL": 285.0,
            "LDL_Cholesterol_mg_dL": 205.0,
            "HDL_Cholesterol_mg_dL": 38.0,
            "Triglycerides_mg_dL": 260.0,
            "ApoB_mg_dL": 145.0,
        }
        res = DiseasePatternEngine.evaluate_condition("dyslipidemia", lipid_data)
        self.assertEqual(res.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Severe Hypercholesterolemia" in p for p in res.patterns_detected))
        self.assertTrue(any("ApoB" in p for p in res.patterns_detected))

    def test_kidney_staging_and_albuminuria(self):
        """Verify eGFR staging and UACR macroalbuminuria detection."""
        ckd_data = {
            "Creatinine_mg_dL": 2.2,
            "eGFR_mL_min_1_73m2": 32.0,
            "Urea_mg_dL": 65.0,
            "UACR_mg_g": 380.0,
        }
        res = DiseasePatternEngine.evaluate_condition("kidney", ckd_data)
        self.assertEqual(res.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("KDIGO Stage G3" in p for p in res.patterns_detected))
        self.assertTrue(any("Macroalbuminuria" in p for p in res.patterns_detected))

    def test_liver_hepatocellular_vs_cholestatic(self):
        """Verify hepatocellular transaminitis vs cholestatic pattern."""
        liver_data = {
            "ALT_U_L": 145.0,
            "AST_U_L": 130.0,
            "ALP_U_L": 175.0,
            "GGT_U_L": 85.0,
            "Bilirubin_mg_dL": 2.1,
            "Albumin_g_dL": 4.1,
        }
        res = DiseasePatternEngine.evaluate_condition("liver", liver_data)
        self.assertEqual(res.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Hepatocellular" in p for p in res.patterns_detected))
        self.assertTrue(any("Cholestatic" in p for p in res.patterns_detected))

    def test_thyroid_tsh_patterns(self):
        """Verify high TSH (hypothyroidism) and low TSH (hyperthyroidism)."""
        hypo = DiseasePatternEngine.evaluate_condition("thyroid", {"TSH_mIU_L": 11.5})
        self.assertEqual(hypo.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Hypothyroidism" in p for p in hypo.patterns_detected))

        hyper = DiseasePatternEngine.evaluate_condition("thyroid", {"TSH_mIU_L": 0.04})
        self.assertEqual(hyper.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Hyperthyroidism" in p for p in hyper.patterns_detected))

    def test_vitamin_d_deficiency(self):
        """Verify severe vitamin D deficiency with hypocalcemia."""
        vit_d_data = {"Vitamin_D_ng_mL": 8.5, "Calcium_mg_dL": 8.1}
        res = DiseasePatternEngine.evaluate_condition("vitamin_d", vit_d_data)
        self.assertEqual(res.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Severe Vitamin D Deficiency" in p for p in res.patterns_detected))

    def test_critical_alert_priority_normal_vitamin_d_with_critical_calcium(self):
        """Verify critical calcium alert elevates status to ABNORMAL_PATTERN even when vitamin D is optimal."""
        # Vitamin D is normal (35.0 ng/mL), but Calcium is panic low (6.0 mg/dL < 6.5 mg/dL)
        data = {"Vitamin_D_ng_mL": 35.0, "Calcium_mg_dL": 6.0}
        res = DiseasePatternEngine.evaluate_condition("vitamin_d", data)
        self.assertEqual(res.status, "ABNORMAL_PATTERN")
        self.assertTrue(len(res.critical_alerts) > 0)
        self.assertTrue(any("CRITICAL" in p for p in res.patterns_detected))
        self.assertTrue(any("Hypocalcemia" in p for p in res.patterns_detected))

    def test_b12_and_folate_deficiency(self):
        """Verify dual deficiency with hyperhomocysteinemia."""
        b12_data = {
            "Vitamin_B12_pg_mL": 150.0,
            "Folate_ng_mL": 3.2,
            "Homocysteine_umol_L": 22.0,
            "MCV_fL": 105.0,
        }
        res = DiseasePatternEngine.evaluate_condition("b12_folate", b12_data)
        self.assertEqual(res.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("B12 Deficiency" in p for p in res.patterns_detected))
        self.assertTrue(any("Folate Deficiency" in p for p in res.patterns_detected))
        self.assertTrue(any("Hyperhomocysteinemia" in p for p in res.patterns_detected))

    def test_electrolyte_derangements(self):
        """Verify electrolyte evaluation and critical potassium alerts."""
        elec_data = {
            "Sodium_mmol_L": 128.0,
            "Potassium_mmol_L": 6.3,
            "Calcium_mg_dL": 8.2,
            "Magnesium_mg_dL": 1.4,
            "Albumin_g_dL": 3.2,
        }
        res = DiseasePatternEngine.evaluate_condition("electrolytes", elec_data)
        self.assertEqual(res.status, "ABNORMAL_PATTERN")
        self.assertTrue(any("Hyponatremia" in p for p in res.patterns_detected))
        self.assertTrue(any("Severe Hyperkalemia" in p for p in res.patterns_detected))

    def test_insufficient_data_missing_required(self):
        """Verify that omitting required biomarkers correctly returns INSUFFICIENT_DATA."""
        res = DiseasePatternEngine.evaluate_condition("diabetes", {"Glucose_mg_dL": 120.0})  # Missing HbA1c
        self.assertEqual(res.status, "INSUFFICIENT_DATA")
        self.assertIn("HbA1c_pct", res.missing_required)

    def test_unknown_condition_handling(self):
        """Verify handling of invalid condition identifier."""
        res = DiseasePatternEngine.evaluate_condition("nonexistent_disease", {})
        self.assertEqual(res.status, "INSUFFICIENT_DATA")


class TestDiseaseExplainer(unittest.TestCase):
    """Test suite for clinical explainer module."""

    def test_all_10_conditions_have_explanations(self):
        """Verify every defined condition panel has complete educational content."""
        for c_id in CONDITION_PANELS.keys():
            exp = DiseaseExplainer.get_explanation(c_id)
            self.assertIn("title", exp)
            self.assertIn("description", exp)
            self.assertIn("why_warrants_attention", exp)
            self.assertIn("potential_complications", exp)
            self.assertIn("general_guidance", exp)
            self.assertIn("consultation_advice", exp)
            self.assertTrue(len(exp["potential_complications"]) > 0)


if __name__ == "__main__":
    unittest.main()
