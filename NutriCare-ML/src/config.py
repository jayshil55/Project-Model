"""
NutriCare ML -- Configuration Constants
All project-wide settings in one place for reproducibility and portability.
"""
import os

# -- Paths (relative to project root) ----------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, "TrainData.xlsx")
DATASET_PATH = DATA_FILE
DATASET_SHEET = "Synthetic_Dataset"
LABELING_RULES_SHEET = "Labeling_Rules"
DATA_DICTIONARY_SHEET = "Data_Dictionary"

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(MODEL_DIR, "nutricare_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# -- Feature / Target -------------------------------------------------
TARGET_COL = "Health_Label"
ID_COL = "Record_ID"
EXCLUDE_COLS = [ID_COL, TARGET_COL]

TARGET_CLASSES = ["GOOD", "MEDIUM", "HARD"]

# -- Training ---------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200
MAX_DEPTH = None  # Let RF decide
MIN_SAMPLES_LEAF = 2

# -- Display ----------------------------------------------------------
SEPARATOR = "=" * 70
