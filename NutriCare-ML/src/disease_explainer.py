"""
NutriCare ML -- Clinical Condition Explainer Module
Author: Senior ML & Health-Data Software Engineer
Date: 2026-10-12

Provides medically grounded, educational explanations for each condition category:
  * Condition overview and physiological role
  * Significance of key biomarkers
  * Why observed patterns warrant attention
  * Potential health risks if an actual condition remains unmanaged
  * Evidence-based general lifestyle, dietary, and preventive guidance
  * Professional healthcare consultation guidance
"""

from typing import Dict, Any


CONDITION_EXPLANATIONS: Dict[str, Dict[str, Any]] = {
    "diabetes": {
        "title": "Abnormal Glucose Regulation & Diabetes Mellitus Screening",
        "description": (
            "Diabetes mellitus encompasses metabolic disorders characterized by persistent hyperglycemia "
            "resulting from defects in insulin secretion, insulin action, or both. Over time, chronic "
            "hyperglycemia leads to systemic vascular endothelial damage and multi-organ dysfunction."
        ),
        "biomarker_roles": {
            "Glucose": "Measures acute circulating sugar; varies with meals, physical activity, and fasting status.",
            "HbA1c": "Reflects the percentage of hemoglobin that is glycated, serving as an index of average glycemia over the preceding 2 to 3 months.",
        },
        "why_warrants_attention": (
            "An elevated glycemic pattern indicates that the body's insulin-mediated glucose disposal is impaired. "
            "Early identification in the prediabetes stage allows effective behavioral and dietary interventions "
            "before irreversible microvascular injury develops."
        ),
        "potential_complications": [
            "Microvascular disease: Diabetic retinopathy (vision loss), nephropathy (kidney failure), and neuropathy (nerve damage, loss of protective sensation).",
            "Macrovascular disease: 2- to 4-fold increased risk of coronary artery disease, myocardial infarction, stroke, and peripheral artery disease.",
            "Acute metabolic crises: Diabetic ketoacidosis (DKA) or hyperosmolar hyperglycemic state (HHS).",
        ],
        "general_guidance": [
            "Adopt dietary patterns rich in whole grains, fiber, lean proteins, and unsaturated fats while minimizing refined carbohydrates and sugar-sweetened beverages (ADA Standards of Care).",
            "Engage in at least 150 minutes of moderate-intensity aerobic physical activity weekly, combined with resistance training.",
            "Maintain moderate weight loss (5% to 7% of body weight) in individuals with excess adiposity.",
        ],
        "consultation_advice": (
            "Consult a primary care physician or endocrinologist for formal diagnostic testing, including a confirmed "
            "fasting plasma glucose or 2-hour oral glucose tolerance test. Never initiate medication without medical supervision."
        ),
    },
    "anemia": {
        "title": "Anemia Patterns (Erythrocyte Deficiency & Morphology)",
        "description": (
            "Anemia is defined as a reduction in circulating hemoglobin concentration, red blood cell mass, or hematocrit "
            "below physiological thresholds, diminishing the blood's oxygen-carrying capacity to peripheral tissues."
        ),
        "biomarker_roles": {
            "Hemoglobin": "The primary iron-containing metalloprotein responsible for transporting oxygen throughout the body.",
            "MCV (Mean Corpuscular Volume)": "Quantifies average red cell volume; differentiates microcytic (<80 fL), normocytic (80-100 fL), and macrocytic (>100 fL) etiologies.",
            "RDW (Red Cell Distribution Width)": "Measures erythrocyte size variation (anisocytosis); elevated in mixed or evolving nutritional deficiencies.",
        },
        "why_warrants_attention": (
            "Anemia causes tissue hypoxia, manifesting as chronic fatigue, exercise intolerance, exertional dyspnea, "
            "dizziness, and cognitive fog. Severe anemia increases cardiac workload and can precipitate high-output cardiac failure."
        ),
        "potential_complications": [
            "Compensatory tachycardia, cardiomegaly, and exacerbated underlying heart disease.",
            "Impaired cognitive performance, chronic exhaustion, and diminished quality of life.",
            "In pregnancy: increased risk of preterm labor, low birth weight, and maternal morbidity.",
        ],
        "general_guidance": [
            "Identify the specific underlying etiology before self-administering supplements; empiric iron or vitamin therapy can mask underlying pathologies.",
            "Ensure a nutrient-dense diet containing adequate dietary iron, folate, and vitamin B12.",
        ],
        "consultation_advice": (
            "Seek evaluation by a healthcare provider or hematologist for comprehensive blood smear examination, "
            "reticulocyte count, iron studies, and gastrointestinal source evaluation if iron deficiency is identified in adult men or postmenopausal women."
        ),
    },
    "iron_deficiency": {
        "title": "Iron Deficiency Patterns & Storage Depletion",
        "description": (
            "Iron is an indispensable trace element required for hemoglobin synthesis, myoglobin function, cellular respiration, "
            "and enzymatic activities. Iron deficiency progresses sequentially from storage depletion (low ferritin) to iron-deficient "
            "erythropoiesis and ultimately microcytic hypochromic iron deficiency anemia (IDA)."
        ),
        "biomarker_roles": {
            "Ferritin": "Intracellular iron-storage protein; the most sensitive laboratory index of total-body iron stores.",
            "CRP": "Inflammatory marker required to interpret ferritin, as inflammation falsely elevates ferritin synthesis.",
        },
        "why_warrants_attention": (
            "Iron depletion impairs cellular mitochondrial enzyme synthesis and oxygen delivery even before overt anemia develops, "
            "contributing to restless legs syndrome, hair loss, brittle nails (koilonychia), pica, and fatigue."
        ),
        "potential_complications": [
            "Severe microcytic hypochromic anemia requiring blood transfusion or intravenous iron therapy.",
            "In adults, unexplained iron deficiency frequently signals occult gastrointestinal blood loss requiring endoscopic evaluation.",
        ],
        "general_guidance": [
            "Consume iron-rich foods: heme iron (poultry, fish, lean meats) and non-heme iron (legumes, dark leafy greens, fortified cereals).",
            "Pair non-heme iron sources with Vitamin C (ascorbic acid) to enhance absorption; avoid drinking coffee or tea simultaneously with meals.",
        ],
        "consultation_advice": (
            "Consult a physician to investigate potential sources of chronic blood loss (e.g., gastrointestinal, gynecological) "
            "and determine the appropriate dosage and formulation of therapeutic iron replacement."
        ),
    },
    "dyslipidemia": {
        "title": "Cholesterol & Lipid Abnormalities (Atherogenic Dyslipidemia)",
        "description": (
            "Dyslipidemia encompasses elevated circulating concentrations of total cholesterol, low-density lipoprotein (LDL), "
            "triglycerides, apolipoprotein B, or reduced high-density lipoprotein (HDL). These lipid derangements drive the formation "
            "and progression of fibrofatty atheromatous plaques within arterial walls."
        ),
        "biomarker_roles": {
            "LDL Cholesterol": "Primary atherogenic target; enters arterial intima, undergoes oxidation, and initiates plaque formation.",
            "HDL Cholesterol": "Mediates reverse cholesterol transport, removing cholesterol from peripheral vessels back to the liver.",
            "Triglycerides": "Circulating neutral fats; elevated levels correlate with metabolic syndrome and acute pancreatitis risk (>500 mg/dL).",
            "ApoB & Lp(a)": "ApoB measures total atherogenic particle count; Lipoprotein(a) is an independent genetically determined risk factor.",
        },
        "why_warrants_attention": (
            "Atherosclerosis develops asymptomatically over decades. Uncontrolled dyslipidemia is a leading modifiable contributor "
            "to premature ischemic cardiovascular events."
        ),
        "potential_complications": [
            "Coronary artery disease, angina, and acute myocardial infarction (heart attack).",
            "Cerebrovascular disease, carotid stenosis, and ischemic stroke.",
            "Peripheral artery disease (PAD) and acute hypertriglyceridemic pancreatitis.",
        ],
        "general_guidance": [
            "Adopt heart-healthy dietary patterns (Mediterranean or DASH diets) emphasizing soluble fiber, plant sterols, and healthy unsaturated fats while eliminating trans fats and reducing saturated fats to <7% of calories.",
            "Engage in regular moderate-to-vigorous aerobic exercise to raise HDL and lower triglycerides.",
            "Avoid all forms of tobacco smoking, which oxidizes LDL and accelerates endothelial injury.",
        ],
        "consultation_advice": (
            "Discuss cardiovascular risk calculation (10-year ASCVD score) with a healthcare professional to assess whether lipid-lowering "
            "pharmacotherapy (e.g., statin therapy) is recommended in accordance with AHA/ACC guidelines."
        ),
    },
    "kidney": {
        "title": "Kidney Function & Renal Filtration Indicators",
        "description": (
            "The kidneys filter metabolic waste products, maintain systemic electrolyte and fluid balance, regulate blood pressure, "
            "and synthesize hormones (erythropoietin, active calcitriol). Impaired glomerular filtration or microvascular leak causes "
            "waste accumulation (uremia) and systemic morbidity."
        ),
        "biomarker_roles": {
            "Creatinine": "Endogenous muscle breakdown metabolite cleared almost exclusively by glomerular filtration.",
            "eGFR": "Estimated glomerular filtration rate; categorizes functional renal stages (KDIGO G1 to G5).",
            "Urea": "Nitrogenous byproduct of protein catabolism; rises in renal dysfunction, dehydration, and catabolic states.",
            "UACR": "Urine albumin-to-creatinine ratio; gold standard for detecting early glomerular microvascular permeability (albuminuria).",
        },
        "why_warrants_attention": (
            "Kidney disease is notoriously silent until advanced stages. Early detection of reduced eGFR or microalbuminuria enables "
            "renoprotective therapies (ACE inhibitors, ARBs, SGLT2 inhibitors) that can halt or delay progression to dialysis."
        ),
        "potential_complications": [
            "Progression to End-Stage Renal Disease (ESRD) requiring long-term hemodialysis or renal transplantation.",
            "Secondary hypertension, fluid retention, volume overload, and pulmonary edema.",
            "Metabolic bone disease (renal osteodystrophy), anemia of chronic kidney disease, and severe electrolyte disturbances.",
        ],
        "general_guidance": [
            "Maintain strict blood pressure control (<130/80 mmHg in proteinuric CKD) and glycemic control in diabetic patients.",
            "Avoid nephrotoxic agents, particularly chronic nonsteroidal anti-inflammatory drugs (NSAIDs) and unvetted herbal supplements.",
            "Follow physician-recommended dietary sodium restriction (<2,000 mg/day).",
        ],
        "consultation_advice": (
            "Consult a nephrologist or primary care physician for confirmatory testing over 90 days, renal ultrasound imaging, and comprehensive medication review."
        ),
    },
    "liver": {
        "title": "Liver Chemistries & Hepatobiliary Function",
        "description": (
            "The liver performs carbohydrate, lipid, and protein metabolism, drug detoxification, bile synthesis, and coagulation factor "
            "production. Hepatic injury may manifest as hepatocellular necrosis (aminotransferase leak), cholestasis (biliary flow obstruction), "
            "or synthetic failure."
        ),
        "biomarker_roles": {
            "ALT & AST": "Intracellular enzymes released into the bloodstream upon hepatocyte membrane disruption or necrosis.",
            "ALP & GGT": "Enzymes located on biliary canalicular membranes; concordant elevation confirms hepatobiliary cholestasis.",
            "Bilirubin": "Heme breakdown product; elevated levels indicate biliary obstruction, hemolysis, or impaired hepatic conjugation.",
            "Albumin": "Primary plasma protein synthesized exclusively by hepatocytes; reflects long-term synthetic function.",
        },
        "why_warrants_attention": (
            "Persistent hepatocellular inflammation leads to progressive hepatic fibrosis, cirrhosis, portal hypertension, and liver failure. "
            "Many causes (viral hepatitis, non-alcoholic fatty liver disease, alcohol, drug toxicity) are treatable if detected early."
        ),
        "potential_complications": [
            "Hepatic fibrosis progressing to decompensated cirrhosis (ascites, esophageal variceal bleeding, hepatic encephalopathy).",
            "Increased risk of hepatocellular carcinoma (liver cancer).",
            "Coagulopathy and spontaneous bacterial peritonitis.",
        ],
        "general_guidance": [
            "Limit or eliminate alcohol consumption to reduce toxic oxidative stress on hepatocytes.",
            "Adopt weight management and aerobic exercise strategies for metabolic dysfunction-associated steatotic liver disease (MASLD).",
            "Review all medications and supplements with a healthcare provider to prevent drug-induced liver injury (DILI).",
        ],
        "consultation_advice": (
            "Consult a gastroenterologist or hepatologist for viral serologies (Hepatitis B & C), abdominal ultrasound or elastography (FibroScan), and directed etiologic evaluation."
        ),
    },
    "thyroid": {
        "title": "Thyroid-Stimulating Hormone (TSH) & Endocrine Homeostasis",
        "description": (
            "The thyroid gland synthesizes thyroxine (T4) and triiodothyronine (T3), hormones that govern basal metabolic rate, cardiac "
            "chronotropy, thermogenesis, and neural function. The anterior pituitary gland regulates thyroid activity through negative feedback "
            "via Thyroid-Stimulating Hormone (TSH)."
        ),
        "biomarker_roles": {
            "TSH": "Pituitary regulatory hormone; rises when circulating thyroid hormones are deficient (hypothyroidism) and falls when thyroid hormones are excessive (hyperthyroidism).",
        },
        "why_warrants_attention": (
            "Thyroid disorders alter global metabolic regulation, producing profound changes in energy levels, cardiovascular hemodynamics, "
            "lipid metabolism, mood, and body temperature regulation."
        ),
        "potential_complications": [
            "Unmanaged Hypothyroidism: Dyslipidemia, pericardial effusion, bradycardia, cognitive slowing, depression, and rare life-threatening myxedema coma.",
            "Unmanaged Hyperthyroidism: Atrial fibrillation, congestive heart failure, osteoporosis/bone demineralization, and thyrotoxic crisis (thyroid storm).",
        ],
        "general_guidance": [
            "Maintain balanced dietary iodine intake without excessive supplementation (e.g. avoid high-dose kelp supplements which can precipitate thyroid dysfunction).",
            "Be aware that biotin supplements can cause laboratory assay interference in TSH testing; discontinue biotin several days prior to thyroid blood tests.",
        ],
        "consultation_advice": (
            "Seek evaluation by an endocrinologist or primary care physician for confirmatory testing including Free T4, Free T3, and anti-TPO thyroid autoantibodies."
        ),
    },
    "vitamin_d": {
        "title": "25-Hydroxy Vitamin D & Skeletal Homeostasis",
        "description": (
            "Vitamin D functions as a prohormone essential for intestinal calcium and phosphorus absorption, skeletal mineralization, and "
            "immunological regulation. 25-hydroxyvitamin D [25(OH)D] is the primary circulating storage form used for clinical assessment."
        ),
        "biomarker_roles": {
            "Vitamin D [25(OH)D]": "Standard circulating biomarker reflecting total dermal synthesis and dietary/supplemental intake.",
            "Calcium & ALP": "Secondary biomarkers reflecting downstream parathyroid compensation and bone turnover.",
        },
        "why_warrants_attention": (
            "Vitamin D deficiency causes compensatory secondary hyperparathyroidism, mobilizing calcium from bone and producing progressive bone loss."
        ),
        "potential_complications": [
            "Osteomalacia (defective bone matrix mineralization causing diffuse bone pain and muscle weakness) in adults, or rickets in children.",
            "Accelerated osteopenia and osteoporosis, predisposing to fragility fractures.",
            "Increased risk of falls due to proximal muscle weakness.",
        ],
        "general_guidance": [
            "Obtain safe sensible sun exposure and incorporate Vitamin D-rich foods (fatty fish like salmon and mackerel, fortified dairy, egg yolks).",
            "Consider physician-supervised cholecalciferol (Vitamin D3) supplementation when dietary intake and sunlight exposure are insufficient.",
        ],
        "consultation_advice": (
            "Consult a medical provider to determine appropriate individualized repletion dosing (e.g. 50,000 IU weekly vs daily maintenance) and assess bone density (DEXA scan) if indicated."
        ),
    },
    "b12_folate": {
        "title": "Vitamin B12 & Folate (One-Carbon Metabolism & Hematopoiesis)",
        "description": (
            "Vitamin B12 (cobalamin) and Folate (vitamin B9) are essential water-soluble coenzymes required for DNA synthesis, erythrocyte "
            "maturation, and the methylation of homocysteine to methionine. Deficiency impairs rapidly dividing cells and central nervous system myelin integrity."
        ),
        "biomarker_roles": {
            "Vitamin B12": "Essential for methionine synthase and methylmalonyl-CoA mutase; deficiency causes both hematologic and neurological damage.",
            "Folate": "Required for purine and thymidylate synthesis in DNA replication.",
            "Homocysteine": "Metabolite that accumulates when either B12 or folate pathways are impaired.",
        },
        "why_warrants_attention": (
            "Crucially, Vitamin B12 deficiency can cause irreversible neurological damage (subacute combined degeneration of the spinal cord, "
            "peripheral neuropathy, memory impairment) even in the absence of severe anemia."
        ),
        "potential_complications": [
            "Permanent peripheral neuropathy, sensory ataxia, spasticity, and cognitive decline if B12 deficiency is prolonged.",
            "Megaloblastic anemia and pancytopenia.",
            "Neural tube defects in pregnancy associated with periconceptional maternal folate deficiency.",
        ],
        "general_guidance": [
            "Individuals following strict vegetarian or vegan diets should routinely consume fortified foods or daily Vitamin B12 supplements.",
            "Never self-treat suspected deficiency with folate alone without ruling out B12 deficiency, as folate can mask the hematologic warning signs of progressive B12 neurotoxicity.",
        ],
        "consultation_advice": (
            "Consult a physician for confirmatory testing (Methylmalonic acid, intrinsic factor antibodies) and tailored oral or intramuscular cobalamin therapy."
        ),
    },
    "electrolytes": {
        "title": "Serum Electrolytes & Fluid-Osmolar Homeostasis",
        "description": (
            "Electrolytes (sodium, potassium, calcium, magnesium) regulate transmembrane electrical potentials, neuromuscular excitability, "
            "osmotic fluid distribution, and cardiac pacemaker conductivity. Imbalances require rapid clinical recognition."
        ),
        "biomarker_roles": {
            "Sodium": "Primary extracellular cation governing plasma osmolality and fluid shifts.",
            "Potassium": "Major intracellular cation; determines cardiac resting membrane potential.",
            "Calcium": "Essential for myocardial contractility, neuromuscular transmission, and bone structure.",
            "Magnesium": "Enzymatic cofactor; stabilizes the sodium-potassium ATPase pump and cardiac rhythm.",
        },
        "why_warrants_attention": (
            "Acute electrolyte derangements can induce fatal cardiac arrhythmias, seizures, and cerebral edema within hours."
        ),
        "potential_complications": [
            "Hyperkalemia (>6.0 mmol/L): Peaked T waves, heart block, and lethal ventricular fibrillation.",
            "Hypokalemia (<3.0 mmol/L): Ventricular arrhythmias, muscle paralysis, and rhabdomyolysis.",
            "Severe Hyponatremia (<125 mmol/L): Cerebral edema, uncal herniation, seizures, and coma.",
            "Hypocalcemia: Tetany, carpopedal spasm, laryngeal stridor, and prolonged QT interval.",
        ],
        "general_guidance": [
            "Maintain balanced dietary intake and adequate hydration; avoid rapid unguided fluid loading.",
            "Never consume high-dose potassium or magnesium supplements without ongoing laboratory monitoring by a physician.",
        ],
        "consultation_advice": (
            "Severe or critical electrolyte derangements represent potential medical emergencies requiring immediate emergency department evaluation, venous blood confirmation, and ECG monitoring."
        ),
    },
}


class DiseaseExplainer:
    """Provides modular, educational condition explanations."""

    @classmethod
    def get_explanation(cls, condition_id: str) -> Dict[str, Any]:
        return CONDITION_EXPLANATIONS.get(condition_id, {
            "title": "General Health Information",
            "description": "Educational context for this biomarker panel is currently unavailable.",
            "biomarker_roles": {},
            "why_warrants_attention": "Any abnormal biomarker pattern warrants clinical assessment.",
            "potential_complications": [],
            "general_guidance": ["Consult a qualified medical provider."],
            "consultation_advice": "Consult a healthcare professional for clinical guidance.",
        })
