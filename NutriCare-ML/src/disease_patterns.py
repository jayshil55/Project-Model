"""
NutriCare ML -- Disease Pattern Recognition Engine
Author: Senior ML & Health-Data Software Engineer
Date: 2026-10-12

Rule-based clinical pattern recognition engine for 10 condition categories.
Evaluates validated biomarker measurements against established clinical consensus.
This module is strictly educational and separated from the 3-class ML model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from src.biomarker_validation import BiomarkerValidator, BiomarkerValidationResult, BIOMARKER_SPECS


@dataclass
class ConditionAssessment:
    condition_id: str
    condition_name: str
    status: str  # 'NORMAL', 'BORDERLINE', 'ABNORMAL_PATTERN', 'CRITICAL_ALERT', 'INSUFFICIENT_DATA'
    patterns_detected: List[str] = field(default_factory=list)
    findings_detail: List[str] = field(default_factory=list)
    biomarkers_evaluated: Dict[str, Any] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    clinical_caveats: List[str] = field(default_factory=list)
    unmeasured_clinical_context: List[str] = field(default_factory=list)
    critical_alerts: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Definition of Condition Panels
# Maps each condition to required biomarkers, optional biomarkers, and missing tests
# ---------------------------------------------------------------------------

CONDITION_PANELS: Dict[str, Dict[str, Any]] = {
    "diabetes": {
        "id": "diabetes",
        "name": "Diabetes-Related Screening & Glycemic Patterns",
        "required": ["Glucose_mg_dL", "HbA1c_pct"],
        "optional": [],
        "missing_context": [
            "Confirmed 8-hour fasting status (synthetic glucose fasting status is unrecorded)",
            "2-hour Oral Glucose Tolerance Test (OGTT)",
            "Fasting C-peptide and Insulin levels",
            "Islet autoantibodies (Anti-GAD, IA-2, ZnT8) to distinguish Type 1 from Type 2 diabetes",
        ],
    },
    "anemia": {
        "id": "anemia",
        "name": "Anemia Patterns (Morphologic Classification)",
        "required": ["Hemoglobin_g_dL", "MCV_fL"],
        "optional": ["RBC_10^12_L", "MCH_pg", "MCHC_g_dL", "RDW_pct", "Ferritin_ng_mL", "Vitamin_B12_pg_mL", "Folate_ng_mL"],
        "missing_context": [
            "Patient biological sex and pregnancy status (hemoglobin cutoffs differ by sex)",
            "Reticulocyte count (to assess bone marrow erythropoietic response)",
            "Peripheral blood smear microscopy for abnormal erythrocyte morphology",
            "Hemoglobin electrophoresis (for thalassemia / sickle cell traits)",
        ],
    },
    "iron_deficiency": {
        "id": "iron_deficiency",
        "name": "Iron Deficiency Patterns",
        "required": ["Ferritin_ng_mL"],
        "optional": ["Hemoglobin_g_dL", "MCV_fL", "MCH_pg", "RDW_pct", "CRP_mg_L"],
        "missing_context": [
            "Serum iron and Total Iron Binding Capacity (TIBC)",
            "Transferrin Saturation percentage (TSAT = Iron / TIBC * 100, diagnostic of IDA if < 20%)",
            "Soluble transferrin receptor (sTfR) to differentiate iron deficiency from anemia of chronic disease",
        ],
    },
    "dyslipidemia": {
        "id": "dyslipidemia",
        "name": "Cholesterol & Lipid Abnormalities (Dyslipidemia)",
        "required": ["Total_Cholesterol_mg_dL", "LDL_Cholesterol_mg_dL", "HDL_Cholesterol_mg_dL", "Triglycerides_mg_dL"],
        "optional": ["ApoB_mg_dL", "Lipoprotein_a_mg_dL"],
        "missing_context": [
            "Fasting vs non-fasting state (triglycerides elevate significantly postprandially)",
            "Patient age, systolic blood pressure, smoking status, and diabetic status (required for 10-year ASCVD risk calculation)",
            "High-sensitivity Troponin and Coronary Artery Calcium (CAC) score",
        ],
    },
    "kidney": {
        "id": "kidney",
        "name": "Kidney Function Abnormalities",
        "required": ["Creatinine_mg_dL", "eGFR_mL_min_1_73m2"],
        "optional": ["Urea_mg_dL", "UACR_mg_g"],
        "missing_context": [
            "Chronicity verification: Diagnosis of CKD requires persistence for >= 3 months (cross-sectional lab cannot distinguish AKI from CKD)",
            "Urinalysis with microscopic examination for sediment and cellular casts",
            "Renal ultrasound imaging",
            "Serum Cystatin C for confirmatory eGFR assessment",
        ],
    },
    "liver": {
        "id": "liver",
        "name": "Liver-Related Abnormalities",
        "required": ["ALT_U_L", "AST_U_L"],
        "optional": ["ALP_U_L", "Bilirubin_mg_dL", "GGT_U_L", "Albumin_g_dL"],
        "missing_context": [
            "Viral hepatitis serologies (HBsAg, Anti-HCV, Anti-HAV)",
            "Hepatic ultrasonography / transient elastography (FibroScan) for steatosis and fibrosis",
            "Prothrombin Time / International Normalized Ratio (PT/INR) for acute synthetic function",
            "Alcohol intake history and medication review (acetaminophen, statins, herbal supplements)",
        ],
    },
    "thyroid": {
        "id": "thyroid",
        "name": "Thyroid Dysfunction Indicators",
        "required": ["TSH_mIU_L"],
        "optional": [],
        "missing_context": [
            "Free Thyroxine (FT4) and Free Triiodothyronine (FT3) -- completely missing from dataset",
            "Without FT4, it is clinically impossible to distinguish subclinical from overt hypothyroidism/hyperthyroidism",
            "Thyroid peroxidase antibodies (Anti-TPO) and Thyroglobulin antibodies (TgAb) for autoimmune Hashimoto's thyroiditis",
            "TSH receptor antibodies (TRAb) for Graves' disease",
        ],
    },
    "vitamin_d": {
        "id": "vitamin_d",
        "name": "Vitamin D Deficiency Patterns",
        "required": ["Vitamin_D_ng_mL"],
        "optional": ["Calcium_mg_dL", "ALP_U_L"],
        "missing_context": [
            "Intact Parathyroid Hormone (PTH) to evaluate secondary hyperparathyroidism",
            "24-hour urinary calcium excretion",
            "Dual-energy X-ray Absorptiometry (DEXA) bone mineral density scan",
        ],
    },
    "b12_folate": {
        "id": "b12_folate",
        "name": "Vitamin B12 and Folate Deficiency Patterns",
        "required": ["Vitamin_B12_pg_mL", "Folate_ng_mL"],
        "optional": ["Homocysteine_umol_L", "MCV_fL", "Hemoglobin_g_dL", "RDW_pct"],
        "missing_context": [
            "Serum Methylmalonic Acid (MMA) -- essential specific functional marker for tissue B12 deficiency",
            "Anti-intrinsic factor antibodies and Anti-parietal cell antibodies (for pernicious anemia)",
            "Red Blood Cell (RBC) folate level (more stable tissue folate indicator than serum folate)",
        ],
    },
    "electrolytes": {
        "id": "electrolytes",
        "name": "Electrolyte Abnormalities",
        "required": ["Sodium_mmol_L", "Potassium_mmol_L", "Calcium_mg_dL", "Magnesium_mg_dL"],
        "optional": ["Albumin_g_dL"],
        "missing_context": [
            "Sample hemolysis status: in-vitro hemolysis during blood draw falsely elevates potassium (pseudohyperkalemia)",
            "Ionized calcium measurement (free physiologically active calcium)",
            "Serum and urine osmolality, urine sodium",
            "12-lead Electrocardiogram (ECG) for cardiac conduction effects of potassium/calcium derangements",
        ],
    },
}


class DiseasePatternEngine:
    """
    Evaluates validated biomarker inputs against evidence-based pattern rules.
    """

    @classmethod
    def evaluate_condition(cls, condition_id: str, inputs: Dict[str, Any]) -> ConditionAssessment:
        if condition_id not in CONDITION_PANELS:
            return ConditionAssessment(
                condition_id=condition_id,
                condition_name="Unknown Condition",
                status="INSUFFICIENT_DATA",
                patterns_detected=["Error: Condition identifier not recognized."],
            )

        panel_meta = CONDITION_PANELS[condition_id]
        required_fields = panel_meta["required"]

        # Validate inputs
        validation_results, panel_errors = BiomarkerValidator.validate_panel(inputs, required_fields)

        # Check for blocking errors
        if panel_errors:
            return ConditionAssessment(
                condition_id=condition_id,
                condition_name=panel_meta["name"],
                status="INSUFFICIENT_DATA",
                patterns_detected=["Insufficient or invalid data for assessment."],
                findings_detail=panel_errors,
                biomarkers_evaluated={k: v.cleaned_value for k, v in validation_results.items()},
                missing_required=[r for r in required_fields if r not in inputs or validation_results[r].is_missing],
                clinical_caveats=panel_meta["missing_context"],
                unmeasured_clinical_context=panel_meta["missing_context"],
            )

        # Dispatch to specific condition evaluator
        eval_method = getattr(cls, f"_eval_{condition_id}", None)
        if eval_method:
            cleaned_vals = {k: v.cleaned_value for k, v in validation_results.items() if v.cleaned_value is not None}
            assessment = eval_method(cleaned_vals, panel_meta)
            assessment.unmeasured_clinical_context = panel_meta["missing_context"]

            # Collect any critical alerts and warnings from biomarker validation
            crit_issues = []
            warn_issues = []
            for b_res in validation_results.values():
                for issue in b_res.issues:
                    if issue.severity == "CRITICAL":
                        crit_issues.append(issue.message)
                    elif issue.severity in ("UNIT_MISMATCH", "WARNING"):
                        warn_issues.append(issue.message)

            assessment.critical_alerts.extend(crit_issues)
            assessment.validation_warnings.extend(warn_issues)

            # CRITICAL ALERT OVERRIDE:
            # A critical lab alert must NEVER be hidden by a 'NORMAL' or 'BORDERLINE' condition status.
            if assessment.critical_alerts:
                if assessment.status in ("NORMAL", "BORDERLINE"):
                    assessment.status = "ABNORMAL_PATTERN"
                for alert_msg in assessment.critical_alerts:
                    if not any(alert_msg in p for p in assessment.patterns_detected):
                        assessment.patterns_detected.insert(0, f"CRITICAL ALERT: {alert_msg}")

            return assessment

        return ConditionAssessment(
            condition_id=condition_id,
            condition_name=panel_meta["name"],
            status="INSUFFICIENT_DATA",
            patterns_detected=["No evaluation rule implemented for this condition."],
        )

    # -----------------------------------------------------------------------
    # 1. Diabetes / Glycemic Pattern Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_diabetes(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        gluc = vals.get("Glucose_mg_dL")
        a1c = vals.get("HbA1c_pct")
        patterns = []
        details = []

        is_diabetic_pattern = False
        is_prediabetic_pattern = False

        # ADA Criteria Evaluation
        if a1c >= 6.5 or gluc >= 200.0 or (gluc >= 180.0):
            is_diabetic_pattern = True
            patterns.append("Elevated Glycemic Pattern (Diabetic Range)")
            details.append(f"HbA1c of {a1c:.2f}% meets/exceeds ADA threshold (>= 6.5%) and/or glucose is markedly elevated ({gluc:.1f} mg/dL).")
        elif (5.7 <= a1c < 6.5) or (140.0 <= gluc < 200.0) or (100.0 <= gluc < 126.0):
            is_prediabetic_pattern = True
            patterns.append("Impaired Glycemic Regulation (Prediabetes Range)")
            details.append(f"HbA1c ({a1c:.2f}%) falls in the prediabetes range (5.7% - 6.4%) and/or glucose ({gluc:.1f} mg/dL) indicates impaired regulation.")
        else:
            patterns.append("Normal Glycemic Indicators")
            details.append(f"Both HbA1c ({a1c:.2f}%) and random glucose ({gluc:.1f} mg/dL) are within standard non-diabetic reference ranges.")

        # Discrepancy check
        if (a1c >= 6.5 and gluc < 100.0) or (a1c < 5.7 and gluc >= 180.0):
            details.append("Discrepancy observed between acute glucose and chronic HbA1c. Recommend re-testing and evaluating red cell turnover / fasting status.")

        status = "ABNORMAL_PATTERN" if is_diabetic_pattern else ("BORDERLINE" if is_prediabetic_pattern else "NORMAL")

        caveats = [
            "Fasting status is not recorded in this synthetic dataset. High glucose could be physiologic postprandial spike.",
            "HbA1c can be falsely altered by hemolytic anemia, hemoglobinopathies, or chronic kidney disease.",
            "A clinical diagnosis requires confirmatory repeat testing on a separate day.",
        ]

        return ConditionAssessment(
            condition_id="diabetes",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 2. Anemia Morphologic Pattern Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_anemia(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        hb = vals.get("Hemoglobin_g_dL")
        mcv = vals.get("MCV_fL")
        rdw = vals.get("RDW_pct")
        ferritin = vals.get("Ferritin_ng_mL")
        b12 = vals.get("Vitamin_B12_pg_mL")
        folate = vals.get("Folate_ng_mL")

        patterns = []
        details = []

        is_anemic = hb < 12.0  # Conservative cutoff

        if not is_anemic:
            patterns.append("Hemoglobin Within Non-Anemic Range")
            details.append(f"Hemoglobin ({hb:.1f} g/dL) is above standard anemia screening cutoff.")
            status = "NORMAL"
        else:
            status = "ABNORMAL_PATTERN"
            if hb < 10.0:
                patterns.append("Moderate-to-Severe Anemia Pattern")
                details.append(f"Hemoglobin is significantly reduced ({hb:.1f} g/dL < 10.0 g/dL).")
            else:
                patterns.append("Mild Anemia Pattern")
                details.append(f"Hemoglobin is mildly reduced ({hb:.1f} g/dL).")

            # Morphologic classification by MCV
            if mcv < 80.0:
                patterns.append("Microcytic Erythrocyte Morphology (Low MCV)")
                details.append(f"MCV is low ({mcv:.1f} fL < 80.0 fL). Differential includes iron deficiency, thalassemia trait, and anemia of chronic disease.")
                if ferritin is not None and ferritin < 30.0:
                    details.append(f"Low ferritin ({ferritin:.1f} ng/mL) strongly supports Iron Deficiency Anemia.")
            elif mcv > 100.0:
                patterns.append("Macrocytic Erythrocyte Morphology (High MCV)")
                details.append(f"MCV is elevated ({mcv:.1f} fL > 100.0 fL). Differential includes B12/folate deficiency, liver dysfunction, reticulocytosis, and medications.")
                if b12 is not None and b12 < 200.0:
                    details.append(f"Low Vitamin B12 ({b12:.1f} pg/mL) supports B12-deficiency megaloblastic anemia.")
                if folate is not None and folate < 4.0:
                    details.append(f"Low Folate ({folate:.1f} ng/mL) supports folate-deficiency anemia.")
            else:
                patterns.append("Normocytic Erythrocyte Morphology (Normal MCV)")
                details.append(f"MCV is normal ({mcv:.1f} fL). Differential includes anemia of chronic disease/inflammation, acute blood loss, hemolysis, and mixed deficiencies.")
                if rdw is not None and rdw > 15.0:
                    details.append(f"Elevated RDW ({rdw:.1f}%) with normocytic MCV suggests possible mixed nutritional deficiency (e.g., dual iron and B12 deficiency).")

        caveats = [
            "Biological sex and pregnancy status are unrecorded; WHO anemia cutoffs differ (<12.0 g/dL female, <13.0 g/dL male, <11.0 g/dL pregnant).",
            "Etiology cannot be diagnosed without reticulocyte count, iron studies, and peripheral blood smear review.",
        ]

        return ConditionAssessment(
            condition_id="anemia",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 3. Iron Deficiency Pattern Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_iron_deficiency(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        ferritin = vals.get("Ferritin_ng_mL")
        hb = vals.get("Hemoglobin_g_dL")
        mcv = vals.get("MCV_fL")
        rdw = vals.get("RDW_pct")
        crp = vals.get("CRP_mg_L")

        patterns = []
        details = []

        if ferritin < 15.0:
            status = "ABNORMAL_PATTERN"
            patterns.append("Severe Iron Depletion (Absolute Iron Deficiency)")
            details.append(f"Serum ferritin ({ferritin:.1f} ng/mL < 15 ng/mL) indicates severely depleted iron stores.")
        elif ferritin < 30.0:
            status = "ABNORMAL_PATTERN"
            patterns.append("Low Iron Stores (Iron Deficiency)")
            details.append(f"Serum ferritin ({ferritin:.1f} ng/mL < 30 ng/mL) indicates low iron reserves.")
        elif ferritin < 50.0 and crp is not None and crp >= 5.0:
            status = "BORDERLINE"
            patterns.append("Suspected Functional / Inflammatory Iron Deficiency")
            details.append(f"Ferritin ({ferritin:.1f} ng/mL) with elevated CRP ({crp:.1f} mg/L) suggests inflammation may be masking iron deficiency.")
        else:
            status = "NORMAL"
            patterns.append("Adequate Iron Stores by Ferritin")
            details.append(f"Serum ferritin ({ferritin:.1f} ng/mL) is within the adequate reference range.")

        # Anemia concordance
        if status in ("ABNORMAL_PATTERN", "BORDERLINE") and hb is not None and hb < 12.0:
            patterns.append("Concordant Iron Deficiency Anemia")
            details.append(f"Accompanied by reduced hemoglobin ({hb:.1f} g/dL), consistent with clinical Iron Deficiency Anemia.")
            if mcv is not None and mcv < 80.0:
                details.append(f"Microcytosis present (MCV {mcv:.1f} fL).")
            if rdw is not None and rdw > 14.5:
                details.append(f"Elevated RDW ({rdw:.1f}%) reflects anisocytosis characteristic of iron deficiency.")

        caveats = [
            "Ferritin is an acute-phase reactant; elevated CRP can falsely elevate ferritin despite true tissue iron deficiency.",
            "Transferrin saturation (TSAT) and soluble transferrin receptor are required to confirm ambiguous cases.",
        ]

        return ConditionAssessment(
            condition_id="iron_deficiency",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 4. Dyslipidemia / Lipid Pattern Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_dyslipidemia(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        tc = vals.get("Total_Cholesterol_mg_dL")
        ldl = vals.get("LDL_Cholesterol_mg_dL")
        hdl = vals.get("HDL_Cholesterol_mg_dL")
        tg = vals.get("Triglycerides_mg_dL")
        apob = vals.get("ApoB_mg_dL")
        lpa = vals.get("Lipoprotein_a_mg_dL")

        patterns = []
        details = []
        abnormal_count = 0

        # LDL
        if ldl >= 190.0:
            abnormal_count += 1
            patterns.append("Severe Hypercholesterolemia (LDL >= 190 mg/dL)")
            details.append(f"Markedly elevated LDL ({ldl:.1f} mg/dL >= 190 mg/dL). Warrant evaluation for familial hypercholesterolemia.")
        elif ldl >= 160.0:
            abnormal_count += 1
            patterns.append("Elevated LDL Cholesterol")
            details.append(f"Elevated LDL ({ldl:.1f} mg/dL >= 160 mg/dL).")
        elif ldl >= 130.0:
            patterns.append("Borderline High LDL Cholesterol")
            details.append(f"Borderline LDL ({ldl:.1f} mg/dL).")

        # Triglycerides
        if tg >= 500.0:
            abnormal_count += 1
            patterns.append("Severe Hypertriglyceridemia (Pancreatitis Risk)")
            details.append(f"Triglycerides ({tg:.1f} mg/dL >= 500 mg/dL) carry elevated acute pancreatitis risk.")
        elif tg >= 200.0:
            abnormal_count += 1
            patterns.append("Elevated Triglycerides")
            details.append(f"High triglycerides ({tg:.1f} mg/dL >= 200 mg/dL).")
        elif tg >= 150.0:
            patterns.append("Borderline High Triglycerides")
            details.append(f"Borderline triglycerides ({tg:.1f} mg/dL).")

        # HDL
        if hdl < 40.0:
            abnormal_count += 1
            patterns.append("Low HDL Cholesterol (Atherogenic Risk Factor)")
            details.append(f"Low HDL ({hdl:.1f} mg/dL < 40 mg/dL).")

        # Advanced markers
        if apob is not None and apob >= 130.0:
            patterns.append("High Atherogenic Particle Burden (Elevated ApoB)")
            details.append(f"ApoB ({apob:.1f} mg/dL >= 130 mg/dL) indicates high total atherogenic particle count.")
        if lpa is not None and lpa >= 50.0:
            patterns.append("Elevated Lipoprotein(a)")
            details.append(f"Lipoprotein(a) ({lpa:.1f} mg/dL >= 50 mg/dL) is an independent genetic cardiovascular risk factor.")

        if not patterns:
            patterns.append("Desirable Lipid Profile")
            details.append(f"Total Cholesterol ({tc:.1f}), LDL ({ldl:.1f}), HDL ({hdl:.1f}), and Triglycerides ({tg:.1f}) are within desirable limits.")
            status = "NORMAL"
        elif abnormal_count >= 2 or (ldl >= 190.0) or (tg >= 500.0):
            status = "ABNORMAL_PATTERN"
        else:
            status = "BORDERLINE"

        caveats = [
            "Fasting vs non-fasting status significantly alters triglyceride measurements.",
            "Cardiovascular risk assessment requires clinical context (age, sex, smoking, blood pressure) via pooled cohort equations.",
        ]

        return ConditionAssessment(
            condition_id="dyslipidemia",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 5. Kidney Function Pattern Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_kidney(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        creat = vals.get("Creatinine_mg_dL")
        egfr = vals.get("eGFR_mL_min_1_73m2")
        urea = vals.get("Urea_mg_dL")
        uacr = vals.get("UACR_mg_g")

        patterns = []
        details = []

        # eGFR Staging (KDIGO)
        if egfr < 15.0:
            status = "ABNORMAL_PATTERN"
            patterns.append("Kidney Failure Pattern (KDIGO Stage G5, eGFR < 15)")
            details.append(f"Severely reduced eGFR ({egfr:.1f} mL/min/1.73m^2).")
        elif egfr < 30.0:
            status = "ABNORMAL_PATTERN"
            patterns.append("Severely Decreased Kidney Function (KDIGO Stage G4, eGFR 15-29)")
            details.append(f"eGFR is severely reduced ({egfr:.1f} mL/min/1.73m^2).")
        elif egfr < 60.0:
            status = "ABNORMAL_PATTERN"
            patterns.append("Moderately Decreased Kidney Function (KDIGO Stage G3, eGFR 30-59)")
            details.append(f"eGFR is moderately reduced ({egfr:.1f} mL/min/1.73m^2).")
        elif egfr < 90.0:
            status = "BORDERLINE"
            patterns.append("Mildly Decreased eGFR (KDIGO Stage G2, eGFR 60-89)")
            details.append(f"eGFR is mildly reduced ({egfr:.1f} mL/min/1.73m^2). May be normal in older adults if without albuminuria.")
        else:
            status = "NORMAL"
            patterns.append("Normal or High eGFR (KDIGO Stage G1, eGFR >= 90)")
            details.append(f"eGFR is preserved ({egfr:.1f} mL/min/1.73m^2).")

        # Creatinine elevation
        if creat > 1.3:
            details.append(f"Elevated serum creatinine ({creat:.2f} mg/dL > 1.3 mg/dL).")
            if status == "NORMAL":
                status = "BORDERLINE"

        # Albuminuria (UACR)
        if uacr is not None:
            if uacr >= 300.0:
                patterns.append("Severely Increased Albuminuria / Macroalbuminuria (UACR >= 300 mg/g)")
                details.append(f"UACR is markedly elevated ({uacr:.1f} mg/g). Reflects glomerular microvascular damage.")
                status = "ABNORMAL_PATTERN"
            elif uacr >= 30.0:
                patterns.append("Moderately Increased Albuminuria / Microalbuminuria (UACR 30-299 mg/g)")
                details.append(f"UACR is elevated ({uacr:.1f} mg/g). Early sign of diabetic or hypertensive nephropathy.")
                if status == "NORMAL":
                    status = "BORDERLINE"

        # Urea
        if urea is not None and urea > 50.0:
            details.append(f"Elevated serum urea ({urea:.1f} mg/dL). Note: this is direct urea, not BUN.")

        caveats = [
            "KDIGO definition of Chronic Kidney Disease strictly requires abnormality persisting for >= 3 months.",
            "Cross-sectional measurement cannot differentiate Acute Kidney Injury (AKI) or dehydration from CKD.",
            "eGFR in this dataset is calculated from creatinine; body habitus and diet affect serum creatinine.",
        ]

        return ConditionAssessment(
            condition_id="kidney",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 6. Liver Function Pattern Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_liver(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        alt = vals.get("ALT_U_L")
        ast = vals.get("AST_U_L")
        alp = vals.get("ALP_U_L")
        bili = vals.get("Bilirubin_mg_dL")
        ggt = vals.get("GGT_U_L")
        alb = vals.get("Albumin_g_dL")

        patterns = []
        details = []

        is_alt_high = alt > 45.0
        is_ast_high = ast > 40.0
        is_alp_high = alp is not None and alp > 130.0
        is_bili_high = bili is not None and bili > 1.2
        is_alb_low = alb is not None and alb < 3.5

        if alt >= 120.0 or ast >= 120.0:
            patterns.append("Marked Hepatocellular Injury Pattern (ALT/AST >= 3x ULN)")
            details.append(f"Marked aminotransferase elevation (ALT {alt:.1f} U/L, AST {ast:.1f} U/L).")
            status = "ABNORMAL_PATTERN"
        elif is_alt_high or is_ast_high:
            patterns.append("Mild-to-Moderate Hepatocellular Transaminitis")
            details.append(f"Elevated aminotransferases (ALT {alt:.1f} U/L, AST {ast:.1f} U/L).")
            status = "BORDERLINE"
        else:
            status = "NORMAL"

        # Cholestatic check
        if is_alp_high and ggt is not None and ggt > 50.0:
            patterns.append("Cholestatic Injury Pattern (Concordant ALP and GGT Elevation)")
            details.append(f"Elevated ALP ({alp:.1f} U/L) confirmed by elevated GGT ({ggt:.1f} U/L) suggests biliary tract involvement.")
            status = "ABNORMAL_PATTERN"

        # Bilirubin check
        if is_bili_high:
            details.append(f"Elevated total bilirubin ({bili:.2f} mg/dL > 1.2 mg/dL).")
            if bili >= 3.0:
                patterns.append("Clinically Significant Hyperbilirubinemia / Jaundice Risk")
                status = "ABNORMAL_PATTERN"

        # Synthetic function check
        if is_alb_low:
            details.append(f"Reduced serum albumin ({alb:.2f} g/dL < 3.5 g/dL). May indicate impaired hepatic synthetic capacity, malnutrition, or systemic inflammation.")

        if not patterns:
            patterns.append("Normal Liver Enzyme Profile")
            details.append("Aminotransferases, ALP, bilirubin, and albumin are within standard reference ranges.")

        caveats = [
            "AST is non-specific; muscle injury, strenuous exercise, and hemolysis elevate AST.",
            "Etiology (viral hepatitis, alcohol, MASLD/NAFLD, drug injury) requires serologies, history, and imaging.",
        ]

        return ConditionAssessment(
            condition_id="liver",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 7. Thyroid Dysfunction Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_thyroid(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        tsh = vals.get("TSH_mIU_L")
        patterns = []
        details = []

        if tsh > 10.0:
            status = "ABNORMAL_PATTERN"
            patterns.append("Markedly Elevated TSH (Suspected Primary Hypothyroidism)")
            details.append(f"TSH is markedly elevated ({tsh:.2f} mIU/L > 10.0 mIU/L). Highly suggestive of overt primary hypothyroidism.")
        elif tsh > 4.0:
            status = "BORDERLINE"
            patterns.append("Mildly Elevated TSH (Possible Subclinical Hypothyroidism)")
            details.append(f"TSH is mildly elevated ({tsh:.2f} mIU/L > 4.0 mIU/L).")
        elif tsh < 0.1:
            status = "ABNORMAL_PATTERN"
            patterns.append("Suppressed TSH (Suspected Hyperthyroidism / Thyrotoxicosis)")
            details.append(f"TSH is suppressed ({tsh:.2f} mIU/L < 0.10 mIU/L). Suggestive of primary hyperthyroidism or exogenous thyroxine excess.")
        elif tsh < 0.4:
            status = "BORDERLINE"
            patterns.append("Low-Normal / Mildly Suppressed TSH")
            details.append(f"TSH is low ({tsh:.2f} mIU/L < 0.40 mIU/L).")
        else:
            status = "NORMAL"
            patterns.append("Euthyroid TSH Indicator")
            details.append(f"TSH ({tsh:.2f} mIU/L) is within the normal reference interval (0.40 - 4.00 mIU/L).")

        caveats = [
            "CRITICAL: Free T4 (FT4) and Free T3 (FT3) are NOT measured in this dataset.",
            "It is impossible to confirm whether thyroid dysfunction is overt or subclinical without FT4.",
            "Central (pituitary) secondary hypothyroidism cannot be diagnosed from TSH alone.",
        ]

        return ConditionAssessment(
            condition_id="thyroid",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 8. Vitamin D Deficiency Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_vitamin_d(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        vit_d = vals.get("Vitamin_D_ng_mL")
        ca = vals.get("Calcium_mg_dL")
        alp = vals.get("ALP_U_L")

        patterns = []
        details = []

        if vit_d < 12.0:
            status = "ABNORMAL_PATTERN"
            patterns.append("Severe Vitamin D Deficiency (< 12 ng/mL)")
            details.append(f"25(OH)D is severely low ({vit_d:.1f} ng/mL). High risk for impaired bone mineralization (osteomalacia / rickets).")
        elif vit_d < 20.0:
            status = "ABNORMAL_PATTERN"
            patterns.append("Vitamin D Deficiency (< 20 ng/mL)")
            details.append(f"25(OH)D is deficient ({vit_d:.1f} ng/mL < 20 ng/mL by IOM and Endocrine Society criteria).")
        elif vit_d < 30.0:
            status = "BORDERLINE"
            patterns.append("Vitamin D Insufficiency (20-29 ng/mL)")
            details.append(f"25(OH)D is insufficient ({vit_d:.1f} ng/mL).")
        else:
            status = "NORMAL"
            patterns.append("Sufficient Vitamin D Level (>= 30 ng/mL)")
            details.append(f"25(OH)D ({vit_d:.1f} ng/mL) meets optimal sufficiency criteria.")

        # Supporting calcium / ALP evaluation
        if ca is not None:
            if ca < 6.5:
                patterns.append("Severe Hypocalcemia Alert (Critical Calcium < 6.5 mg/dL)")
                details.append(f"Serum calcium is critically low ({ca:.2f} mg/dL < 6.5 mg/dL). High risk of tetany and cardiac conduction impairment.")
                if status == "NORMAL":
                    status = "ABNORMAL_PATTERN"
            elif ca < 8.5:
                patterns.append("Concomitant Hypocalcemia (Low Calcium)")
                details.append(f"Serum calcium is low ({ca:.2f} mg/dL < 8.5 mg/dL). May reflect impaired intestinal calcium absorption secondary to vitamin D deficiency.")
                if status == "NORMAL":
                    status = "BORDERLINE"
            elif ca > 13.0:
                patterns.append("Severe Hypercalcemia Alert (Critical Calcium > 13.0 mg/dL)")
                details.append(f"Serum calcium is dangerously high ({ca:.2f} mg/dL > 13.0 mg/dL). Urgent medical workup required.")
                if status == "NORMAL":
                    status = "ABNORMAL_PATTERN"
            elif ca > 10.5:
                patterns.append("Concomitant Hypercalcemia (Elevated Calcium)")
                details.append(f"Serum calcium is elevated ({ca:.2f} mg/dL > 10.5 mg/dL). Consider primary hyperparathyroidism or vitamin D toxicity.")
                if status == "NORMAL":
                    status = "BORDERLINE"

        if alp is not None and alp > 130.0:
            details.append(f"Concomitant elevated ALP ({alp:.1f} U/L). May reflect compensatory parathyroid bone remodeling.")

        caveats = [
            "Assay methods vary; Parathyroid Hormone (PTH) is needed to evaluate physiologic secondary hyperparathyroidism.",
            "Levels exhibit seasonal variation and depend heavily on geographical latitude and skin pigmentation.",
        ]

        return ConditionAssessment(
            condition_id="vitamin_d",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 9. Vitamin B12 and Folate Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_b12_folate(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        b12 = vals.get("Vitamin_B12_pg_mL")
        folate = vals.get("Folate_ng_mL")
        hcy = vals.get("Homocysteine_umol_L")
        mcv = vals.get("MCV_fL")
        hb = vals.get("Hemoglobin_g_dL")

        patterns = []
        details = []
        abnormal = False

        # B12
        if b12 < 200.0:
            abnormal = True
            patterns.append("Vitamin B12 Deficiency (< 200 pg/mL)")
            details.append(f"Serum B12 is low ({b12:.1f} pg/mL < 200 pg/mL). Neurological and hematologic risk.")
        elif b12 < 300.0:
            patterns.append("Borderline Vitamin B12 Status (200-300 pg/mL)")
            details.append(f"Serum B12 is in the indeterminate borderline range ({b12:.1f} pg/mL).")

        # Folate
        if folate < 4.0:
            abnormal = True
            patterns.append("Serum Folate Deficiency (< 4.0 ng/mL)")
            details.append(f"Serum folate is deficient ({folate:.1f} ng/mL < 4.0 ng/mL).")
        elif folate < 6.0:
            patterns.append("Low-Normal Folate Status")
            details.append(f"Folate is low-normal ({folate:.1f} ng/mL).")

        # Functional marker (Homocysteine)
        if hcy is not None and hcy > 15.0:
            patterns.append("Hyperhomocysteinemia")
            details.append(f"Elevated homocysteine ({hcy:.1f} umol/L) supports functional cellular deficiency in one-carbon methylation pathways.")

        # Macrocytosis
        if mcv is not None and mcv > 100.0:
            details.append(f"Macrocytosis present (MCV {mcv:.1f} fL > 100 fL), consistent with megaloblastic morphology.")

        if not patterns:
            patterns.append("Normal B12 and Folate Concentrations")
            details.append(f"B12 ({b12:.1f} pg/mL) and Folate ({folate:.1f} ng/mL) are adequate.")
            status = "NORMAL"
        elif abnormal:
            status = "ABNORMAL_PATTERN"
        else:
            status = "BORDERLINE"

        caveats = [
            "Serum B12 measures total cobalamin; methylmalonic acid (MMA) is the gold standard for tissue deficiency.",
            "Folate supplementation can mask B12-deficiency anemia while allowing irreversible neuropathy to progress.",
        ]

        return ConditionAssessment(
            condition_id="b12_folate",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )

    # -----------------------------------------------------------------------
    # 10. Electrolyte Abnormalities Evaluator
    # -----------------------------------------------------------------------
    @staticmethod
    def _eval_electrolytes(vals: Dict[str, float], meta: Dict[str, Any]) -> ConditionAssessment:
        na = vals.get("Sodium_mmol_L")
        k = vals.get("Potassium_mmol_L")
        ca = vals.get("Calcium_mg_dL")
        mg = vals.get("Magnesium_mg_dL")
        alb = vals.get("Albumin_g_dL")

        patterns = []
        details = []
        critical_count = 0
        abnormal_count = 0

        # Sodium
        if na < 130.0:
            critical_count += 1
            patterns.append("Clinically Significant Hyponatremia (< 130 mmol/L)")
            details.append(f"Sodium is markedly low ({na:.1f} mmol/L). Cerebral edema and seizure risk.")
        elif na < 135.0:
            abnormal_count += 1
            patterns.append("Mild Hyponatremia (130-134 mmol/L)")
            details.append(f"Sodium is mildly reduced ({na:.1f} mmol/L).")
        elif na > 150.0:
            critical_count += 1
            patterns.append("Clinically Significant Hypernatremia (> 150 mmol/L)")
            details.append(f"Sodium is markedly high ({na:.1f} mmol/L). Cellular dehydration risk.")
        elif na > 145.0:
            abnormal_count += 1
            patterns.append("Mild Hypernatremia (146-150 mmol/L)")
            details.append(f"Sodium is mildly elevated ({na:.1f} mmol/L).")

        # Potassium
        if k < 3.0:
            critical_count += 1
            patterns.append("Severe Hypokalemia (< 3.0 mmol/L) -- Arrhythmia Alert")
            details.append(f"Potassium is dangerously low ({k:.2f} mmol/L). Urgent cardiac monitoring required.")
        elif k < 3.5:
            abnormal_count += 1
            patterns.append("Mild Hypokalemia (3.0-3.4 mmol/L)")
            details.append(f"Potassium is low ({k:.2f} mmol/L).")
        elif k > 6.0:
            critical_count += 1
            patterns.append("Severe Hyperkalemia (> 6.0 mmol/L) -- Arrhythmia Alert")
            details.append(f"Potassium is dangerously high ({k:.2f} mmol/L). Urgent cardiac monitoring and calcium gluconate consideration.")
        elif k > 5.0:
            abnormal_count += 1
            patterns.append("Mild Hyperkalemia (5.1-6.0 mmol/L)")
            details.append(f"Potassium is elevated ({k:.2f} mmol/L).")

        # Calcium (with albumin correction if available)
        effective_ca = ca
        if alb is not None and alb < 4.0:
            effective_ca = ca + 0.8 * (4.0 - alb)
            details.append(f"Measured calcium: {ca:.2f} mg/dL; Albumin-corrected calcium: {effective_ca:.2f} mg/dL.")

        if effective_ca < 8.5:
            abnormal_count += 1
            patterns.append("Hypocalcemia Pattern")
            details.append(f"Corrected calcium ({effective_ca:.2f} mg/dL < 8.5 mg/dL). Neuromuscular excitability risk.")
        elif effective_ca > 10.5:
            abnormal_count += 1
            patterns.append("Hypercalcemia Pattern")
            details.append(f"Corrected calcium ({effective_ca:.2f} mg/dL > 10.5 mg/dL). Hyperparathyroidism / malignancy workup indicated.")

        # Magnesium
        if mg < 1.7:
            abnormal_count += 1
            patterns.append("Hypomagnesemia Pattern")
            details.append(f"Magnesium is low ({mg:.2f} mg/dL < 1.7 mg/dL). Can cause refractory hypokalemia.")
        elif mg > 2.4:
            abnormal_count += 1
            patterns.append("Hypermagnesemia Pattern")
            details.append(f"Magnesium is elevated ({mg:.2f} mg/dL > 2.4 mg/dL).")

        if not patterns:
            patterns.append("Electrolytes Within Normal Limits")
            details.append(f"Sodium ({na:.1f}), Potassium ({k:.2f}), Calcium ({ca:.2f}), and Magnesium ({mg:.2f}) are normal.")
            status = "NORMAL"
        elif critical_count > 0 or abnormal_count >= 2:
            status = "ABNORMAL_PATTERN"
        else:
            status = "BORDERLINE"

        caveats = [
            "In-vitro hemolysis during phlebotomy falsely elevates potassium (pseudohyperkalemia).",
            "Urgent clinical verification via repeat venous sample and 12-lead ECG is required for critical derangements.",
        ]

        return ConditionAssessment(
            condition_id="electrolytes",
            condition_name=meta["name"],
            status=status,
            patterns_detected=patterns,
            findings_detail=details,
            biomarkers_evaluated=vals,
            clinical_caveats=caveats,
        )
