"""
NutriCare ML -- Model Training Module
Builds a scikit-learn Pipeline with median imputation + Random Forest.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

from src.config import (
    RANDOM_SEED, TEST_SIZE, N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF,
    MODEL_DIR, MODEL_PATH, METADATA_PATH, TARGET_CLASSES, TARGET_COL,
    OUTPUT_DIR, SEPARATOR
)


def build_pipeline() -> Pipeline:
    """Create the preprocessing + model pipeline."""
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    return pipeline


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, feature_names: list) -> dict:
    """
    Full training workflow:
    1. Stratified train/test split
    2. Train Random Forest pipeline
    3. Compare with DummyClassifier baseline
    4. Evaluate on held-out test set
    5. Save model + metadata + reports
    Returns a dict of metrics.
    """
    print(f"\n{SEPARATOR}")
    print("MODEL TRAINING")
    print(SEPARATOR)

    # -- 1. Stratified split -------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"\n[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[SPLIT] Test size: {TEST_SIZE} | Random seed: {RANDOM_SEED}")

    # Class counts
    print(f"\n[TRAIN SET] Class distribution:")
    train_dist = y_train.value_counts().sort_index()
    for label, cnt in train_dist.items():
        print(f"   {label:>8s}: {cnt:>4d} ({cnt/len(y_train)*100:.1f}%)")

    print(f"\n[TEST SET] Class distribution:")
    test_dist = y_test.value_counts().sort_index()
    for label, cnt in test_dist.items():
        print(f"   {label:>8s}: {cnt:>4d} ({cnt/len(y_test)*100:.1f}%)")

    # -- 2. Baseline: DummyClassifier ---------------------------------
    print(f"\n{SEPARATOR}")
    print("BASELINE: DummyClassifier (most_frequent)")
    print(SEPARATOR)
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_acc = accuracy_score(y_test, dummy_pred)
    dummy_f1 = f1_score(y_test, dummy_pred, average="macro", zero_division=0)
    print(f"   Dummy Accuracy: {dummy_acc:.4f}")
    print(f"   Dummy Macro-F1: {dummy_f1:.4f}")

    # -- 3. Train Random Forest pipeline ------------------------------
    print(f"\n{SEPARATOR}")
    print("TRAINING: Random Forest Pipeline")
    print(SEPARATOR)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("[INFO] Pipeline trained successfully.")

    # -- 4. Evaluate on test set --------------------------------------
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    clf_report = classification_report(
        y_test, y_pred, target_names=pipeline.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    print(f"\n{SEPARATOR}")
    print("EVALUATION RESULTS (Test Set)")
    print(SEPARATOR)
    print(f"\n   Accuracy:  {acc:.4f}")
    print(f"   Macro-F1:  {macro_f1:.4f}")
    print(f"\n   Improvement over baseline:")
    print(f"      Accuracy: +{(acc - dummy_acc):.4f}")
    print(f"      Macro-F1: +{(macro_f1 - dummy_f1):.4f}")
    print(f"\n   Classification Report:\n{clf_report}")
    print(f"   Confusion Matrix (rows=true, cols=predicted):")
    print(f"   Labels: {list(pipeline.classes_)}")
    for i, row in enumerate(cm):
        print(f"   {pipeline.classes_[i]:>8s}: {[int(x) for x in row]}")

    # -- 5. Save model + metadata -------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[INFO] Model saved to: {MODEL_PATH}")

    metadata = {
        "project": "NutriCare ML -- Blood Biomarker Analysis",
        "model_type": "RandomForestClassifier",
        "n_estimators": N_ESTIMATORS,
        "max_depth": str(MAX_DEPTH),
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "class_weight": "balanced",
        "preprocessing": "SimpleImputer(strategy='median')",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_file": "TrainData.xlsx",
        "dataset_sheet": "Synthetic_Dataset",
        "total_records": len(X),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "input_features": feature_names,
        "feature_count": len(feature_names),
        "target_column": TARGET_COL,
        "target_classes": [str(c) for c in pipeline.classes_],
        "class_distribution": {
            "total": {str(k): int(v) for k, v in y.value_counts().items()},
            "train": {str(k): int(v) for k, v in y_train.value_counts().items()},
            "test": {str(k): int(v) for k, v in y_test.value_counts().items()},
        },
        "evaluation": {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "dummy_accuracy": round(float(dummy_acc), 4),
            "dummy_macro_f1": round(float(dummy_f1), 4),
        },
        "disclaimer": (
            "This is a synthetic educational dataset. Model performance on this data "
            "does NOT establish real-world medical accuracy. Labels are rule-derived "
            "demonstration categories, not clinical diagnoses or validated risk scores."
        ),
        "safety_warning": (
            "The saved model file (nutricare_model.pkl) uses Python pickle format. "
            "Only load this file from a trusted source. Never load untrusted pickle files."
        ),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Metadata saved to: {METADATA_PATH}")

    # -- 6. Save evaluation report + confusion matrix image -----------
    _save_evaluation_report(
        acc, macro_f1, clf_report, cm, pipeline.classes_,
        train_dist, test_dist, dummy_acc, dummy_f1,
        feature_names
    )
    _save_confusion_matrix_image(cm, pipeline.classes_)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "dummy_accuracy": dummy_acc,
        "dummy_macro_f1": dummy_f1,
        "confusion_matrix": cm,
        "classes": list(pipeline.classes_),
        "classification_report": clf_report,
    }
    return metrics


def _save_evaluation_report(
    acc, macro_f1, clf_report, cm, classes,
    train_dist, test_dist, dummy_acc, dummy_f1, feature_names
):
    """Save a readable evaluation report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []
    lines.append(SEPARATOR)
    lines.append("NUTRICARE ML -- MODEL EVALUATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(SEPARATOR)

    lines.append(f"\nMODEL: Random Forest (n_estimators={N_ESTIMATORS}, "
                 f"class_weight=balanced)")
    lines.append(f"PREPROCESSING: Median imputation (fit on training data only)")
    lines.append(f"FEATURES: {len(feature_names)} numeric biomarkers")
    lines.append(f"SPLIT: {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test, "
                 f"stratified, seed={RANDOM_SEED}")

    lines.append(f"\n{'-'*50}")
    lines.append("BASELINE COMPARISON (DummyClassifier -- most_frequent)")
    lines.append(f"{'-'*50}")
    lines.append(f"   Dummy Accuracy:  {dummy_acc:.4f}")
    lines.append(f"   Dummy Macro-F1:  {dummy_f1:.4f}")

    lines.append(f"\n{'-'*50}")
    lines.append("RANDOM FOREST RESULTS (held-out test set)")
    lines.append(f"{'-'*50}")
    lines.append(f"   Accuracy:        {acc:.4f}")
    lines.append(f"   Macro-F1:        {macro_f1:.4f}")
    lines.append(f"   Acc improvement: +{(acc - dummy_acc):.4f} over baseline")
    lines.append(f"   F1 improvement:  +{(macro_f1 - dummy_f1):.4f} over baseline")

    lines.append(f"\n{'-'*50}")
    lines.append("CLASSIFICATION REPORT")
    lines.append(f"{'-'*50}")
    lines.append(clf_report)

    lines.append(f"{'-'*50}")
    lines.append("CONFUSION MATRIX (rows=actual, cols=predicted)")
    lines.append(f"{'-'*50}")
    lines.append(f"   Labels: {list(classes)}")
    for i, row in enumerate(cm):
        lines.append(f"   {classes[i]:>8s}: {[int(x) for x in row]}")

    lines.append(f"\n{'-'*50}")
    lines.append("CLASS DISTRIBUTION")
    lines.append(f"{'-'*50}")
    lines.append("   Training set:")
    for label, cnt in train_dist.items():
        lines.append(f"      {label:>8s}: {cnt}")
    lines.append("   Test set:")
    for label, cnt in test_dist.items():
        lines.append(f"      {label:>8s}: {cnt}")

    lines.append(f"\n{'-'*50}")
    lines.append("STRENGTHS")
    lines.append(f"{'-'*50}")
    lines.append("   * Random Forest substantially outperforms the dummy baseline.")
    lines.append("   * Preprocessing pipeline ensures no data leakage from test set.")
    lines.append("   * Stratified split preserves class proportions in train/test.")
    lines.append("   * class_weight='balanced' compensates for class imbalance.")

    lines.append(f"\n{'-'*50}")
    lines.append("LIMITATIONS")
    lines.append(f"{'-'*50}")
    lines.append("   * Labels are DETERMINISTIC FUNCTIONS of the input features.")
    lines.append("     High accuracy reflects learned rules, NOT medical insight.")
    lines.append("   * HARD class has very few samples (~27 total, ~5-6 in test).")
    lines.append("     HARD precision/recall are noisy and not statistically robust.")
    lines.append("   * This is synthetic data -- no real-world generalization implied.")
    lines.append("   * Glucose fasting status is unspecified; rules are synthetic.")
    lines.append("   * Missing values are imputed with median, which may wash out")
    lines.append("     extreme values that trigger labeling rules.")

    lines.append(f"\n{'-'*50}")
    lines.append("DISCLAIMER")
    lines.append(f"{'-'*50}")
    lines.append("   This is an EDUCATIONAL ML DEMONSTRATION, not a clinical tool.")
    lines.append("   Model predictions represent learned synthetic patterns only.")
    lines.append("   Do NOT use for medical diagnosis, risk assessment, or treatment.")

    lines.append(f"\n{SEPARATOR}")

    path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Evaluation report saved to: {path}")


def _save_confusion_matrix_image(cm, classes):
    """Save a confusion matrix visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5, linecolor="gray",
        cbar_kws={"label": "Count"}
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(
        "NutriCare ML -- Confusion Matrix (Test Set)\n"
        "[Synthetic Data -- Educational Demo Only]",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Confusion matrix image saved to: {path}")
