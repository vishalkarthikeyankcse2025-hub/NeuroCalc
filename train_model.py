"""
NeuroCalc — Final 1M-Row ML Training Pipeline
==============================================

Purpose
-------
Train the NeuroCalc intent-classification model using the 1M quality-fixed
dataset and create a complete, report-ready experiment package.

Main improvement over the previous trainer
-------------------------------------------
The classifier is now MATH-AWARE:

    125 * 20  -> 125 OP_MUL 20
    125 / 20  -> 125 OP_DIV 20
    125 + 20  -> 125 OP_ADD 20
    125 - 20  -> 125 OP_SUB 20
    125 ^ 20  -> 125 OP_POWER 20
    20%       -> 20 OP_PERCENT

The model also uses two complementary TF-IDF feature spaces:
1. word n-grams
2. character n-grams

This is designed to reduce confusion between mathematical operations while
keeping natural-language understanding.

The math engine remains separate: ML identifies intent; deterministic
mathematics computes the final answer.

Run
---
    python train_model.py

The script automatically supports either:
    data/neurocalc_dataset_1m_quality_fixed.csv
or
    data/neurocalc_dataset_1m.csv

Outputs
-------
models/
    neurocalc_intent_model_1m.joblib

training_results/
    graphs/
        01_dataset_intent_distribution.png
        02_train_test_distribution.png
        03_confusion_matrix.png
        04_precision_recall_f1.png
        05_per_intent_performance.png
        06_confidence_distribution.png
        07_correct_vs_incorrect.png
        08_top_features_by_intent.png
        09_top3_accuracy.png
        10_high_confidence_errors.png

    data/
        cleaned_dataset.csv
        train_data.csv
        test_data.csv
        dataset_statistics.json
        intent_distribution.csv
        split_statistics.json

    report/
        classification_report.txt
        classification_report.csv
        confusion_matrix.csv
        test_predictions.csv
        misclassified_examples.csv
        high_confidence_errors.csv
        misclassification_summary.csv
        confidence_statistics.json
        model_configuration.json
        top_features_by_intent.csv
        training_summary.json
        experiment_report.txt
        model_card.md

Note
----
The 1M dataset is synthetic. High validation performance on synthetic data
must not be presented as proof of real-world language understanding.
Use a separate human-written evaluation set for final real-world claims.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent

# New quality-fixed dataset first, old dataset second.
DATASET_CANDIDATES = [
    BASE / "data" / "neurocalc_dataset_1m_quality_fixed.csv",
    BASE / "data" / "neurocalc_dataset_1m.csv",
]

MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "training_results"

GRAPH_DIR = RESULTS_DIR / "graphs"
DATA_RESULTS_DIR = RESULTS_DIR / "data"
REPORT_DIR = RESULTS_DIR / "report"
CHALLENGE_DATASET = BASE / "data" / "neurocalc_challenge_5k.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Word features
WORD_NGRAM_RANGE = (1, 3)
WORD_MIN_DF = 2
WORD_MAX_DF = 0.995
WORD_MAX_FEATURES = 100_000

# Character features
# char_wb focuses on character patterns inside word boundaries and helps
# distinguish compact mathematical/operator forms after OP_* normalization.
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_DF = 0.995
CHAR_MAX_FEATURES = 50_000

LOGREG_SOLVER = "saga"
LOGREG_MAX_ITER = 150
LOGREG_C = 4.0

TOP_FEATURES_PER_CLASS = 15
TOP3_K = 3

for directory in [
    MODEL_DIR,
    RESULTS_DIR,
    GRAPH_DIR,
    DATA_RESULTS_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    print(f"[NeuroCalc] {message}")


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=4, default=str),
        encoding="utf-8",
    )


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def add_math_tokens(text: str) -> str:
    """
    Preserve mathematical operators as explicit ML tokens.

    This directly addresses operation confusion such as:
        + vs * vs / vs - vs ^

    Negative numbers at the beginning of a token are not rewritten as
    subtraction. Binary operators between operands are rewritten.
    """
    s = str(text)

    replacements = {
        "×": " * ",
        "✕": " * ",
        "÷": " / ",
        "−": " - ",
        "–": " - ",
        "√": " sqrt ",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    # Binary operators between operands.
    s = re.sub(
        r"(?<=\d)\s*\+\s*(?=[\d(])",
        " OP_ADD ",
        s,
    )

    s = re.sub(
        r"(?<=\d)\s*-\s*(?=\d|\()",
        " OP_SUB ",
        s,
    )

    s = re.sub(
        r"(?<=\d)\s*\*\s*(?=[\d(])",
        " OP_MUL ",
        s,
    )

    s = re.sub(
        r"(?<=\d)\s*/\s*(?=[\d(])",
        " OP_DIV ",
        s,
    )

    s = re.sub(
        r"(?<=[A-Za-z0-9)\]])\s*\^\s*(?=[A-Za-z0-9(\\[])",
        " OP_POWER ",
        s,
    )

    s = s.replace("%", " OP_PERCENT ")
    s = s.replace("=", " OP_EQUAL ")
    s = s.replace("[", " OP_LBRACKET ")
    s = s.replace("]", " OP_RBRACKET ")

    # If multiplication/division symbols remained because operands contain
    # non-digit symbolic names, preserve them explicitly.
    s = s.replace("*", " OP_MUL ")
    s = s.replace("/", " OP_DIV ")

    # Add a small signature that summarizes which mathematical operators
    # are physically present. This is redundant by design but useful for
    # short calculator queries dominated by symbols.
    signature = []
    for token in [
        "OP_ADD",
        "OP_SUB",
        "OP_MUL",
        "OP_DIV",
        "OP_POWER",
        "OP_PERCENT",
        "OP_EQUAL",
    ]:
        if token in s:
            signature.append(f"SIG_{token}")

    if signature:
        s += " " + " ".join(signature)

    return " ".join(s.split())


def find_dataset() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    formatted = "\n".join(str(p) for p in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Dataset not found. Expected one of:\n" + formatted
    )


def calculate_ece(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """
    Expected Calibration Error using max predicted probability.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)

    ece = 0.0
    n = len(confidences)

    if n == 0:
        return 0.0

    boundaries = np.linspace(0.0, 1.0, bins + 1)

    for i in range(bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]

        if i == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)

        if not mask.any():
            continue

        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        bin_fraction = mask.mean()

        ece += abs(bin_accuracy - bin_confidence) * bin_fraction

    return float(ece)



# ============================================================
# INDEPENDENT CHALLENGE EVALUATION
# ============================================================

def evaluate_independent_challenge(model, challenge_path, out_dir):
    if not challenge_path.exists():
        raise FileNotFoundError(
            f"Challenge dataset not found: {challenge_path}\n"
            "Put neurocalc_challenge_5k.csv into the data folder."
        )

    challenge = pd.read_csv(challenge_path)

    required = {"text", "intent"}
    missing = required - set(challenge.columns)
    if missing:
        raise ValueError(
            f"Challenge dataset missing columns: {sorted(missing)}"
        )

    challenge = challenge[["text", "intent"]].copy()
    challenge["text"] = challenge["text"].astype(str).str.strip()
    challenge["intent"] = challenge["intent"].astype(str).str.strip()

    if challenge["text"].duplicated().any():
        raise ValueError("Challenge dataset contains duplicate text rows.")

    if len(challenge) != 5000:
        raise ValueError(
            f"Expected 5,000 challenge rows, got {len(challenge)}."
        )

    # Exact leakage check against the entire main dataset.
    main_texts = set(df["text"].astype(str))
    overlap = int(challenge["text"].isin(main_texts).sum())

    if overlap:
        raise RuntimeError(
            f"Challenge leakage detected: {overlap} exact texts overlap "
            "with the main dataset."
        )

    challenge_dir = out_dir / "challenge"
    challenge_dir.mkdir(parents=True, exist_ok=True)

    log(f"Independent challenge samples: {len(challenge):,}")
    log(f"Challenge/main exact text overlap: {overlap}")

    pred_start = time.time()
    predictions = model.predict(challenge["text"])
    probabilities = model.predict_proba(challenge["text"])
    pred_time = time.time() - pred_start

    confidence = probabilities.max(axis=1)
    correct = predictions == challenge["intent"].to_numpy()

    classes = model.classes_

    report_dict = classification_report(
        challenge["intent"],
        predictions,
        labels=classes,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        challenge["intent"],
        predictions,
        labels=classes,
        zero_division=0,
    )

    matrix = confusion_matrix(
        challenge["intent"],
        predictions,
        labels=classes,
    )

    prediction_df = pd.DataFrame({
        "text": challenge["text"],
        "actual_intent": challenge["intent"],
        "predicted_intent": predictions,
        "confidence": confidence,
        "correct": correct,
    })

    prediction_df.to_csv(
        challenge_dir / "challenge_predictions.csv",
        index=False,
    )

    errors = prediction_df[~prediction_df["correct"]].copy()
    errors.to_csv(
        challenge_dir / "challenge_misclassified_examples.csv",
        index=False,
    )

    high_conf_errors = errors[errors["confidence"] >= 0.90].copy()
    high_conf_errors.to_csv(
        challenge_dir / "challenge_high_confidence_errors.csv",
        index=False,
    )

    pd.DataFrame(report_dict).T.to_csv(
        challenge_dir / "challenge_classification_report.csv"
    )

    save_text(
        challenge_dir / "challenge_classification_report.txt",
        report_text,
    )

    pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    ).to_csv(
        challenge_dir / "challenge_confusion_matrix.csv"
    )

    misclassification = (
        prediction_df[~prediction_df["correct"]]
        .groupby(
            ["actual_intent", "predicted_intent"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    misclassification.to_csv(
        challenge_dir / "challenge_misclassification_summary.csv",
        index=False,
    )

    top_k = min(3, probabilities.shape[1])
    top_indices = np.argsort(probabilities, axis=1)[:, -top_k:]

    top3_correct = np.array([
        actual in classes[row]
        for actual, row in zip(
            challenge["intent"].to_numpy(),
            top_indices,
        )
    ])

    top3_accuracy = float(top3_correct.mean())
    accuracy = float(correct.mean())

    confidence_stats = {
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "min_confidence": float(np.min(confidence)),
        "max_confidence": float(np.max(confidence)),
        "correct_mean_confidence": float(
            prediction_df.loc[
                prediction_df["correct"],
                "confidence"
            ].mean()
        ),
        "incorrect_mean_confidence": (
            float(
                prediction_df.loc[
                    ~prediction_df["correct"],
                    "confidence"
                ].mean()
            )
            if len(errors)
            else None
        ),
        "high_confidence_errors": int(len(high_conf_errors)),
        "samples_below_0_50": int(np.sum(confidence < 0.50)),
        "samples_below_0_70": int(np.sum(confidence < 0.70)),
        "samples_above_or_equal_0_90": int(
            np.sum(confidence >= 0.90)
        ),
    }

    save_json(
        challenge_dir / "challenge_confidence_statistics.json",
        confidence_stats,
    )

    save_json(
        challenge_dir / "challenge_dataset_statistics.json",
        {
            "rows": int(len(challenge)),
            "intents": int(challenge["intent"].nunique()),
            "rows_per_intent": {
                str(k): int(v)
                for k, v in challenge["intent"].value_counts().sort_index().items()
            },
            "used_for_training": False,
            "exact_text_overlap_with_main_dataset": overlap,
            "synthetic": True,
            "human_written": False,
        },
    )

    # Challenge confusion matrix graph.
    plt.figure(figsize=(17, 15))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Independent Synthetic Challenge — Confusion Matrix")
    plt.xlabel("Predicted Intent")
    plt.ylabel("Actual Intent")
    plt.colorbar()

    ticks = np.arange(len(classes))
    plt.xticks(
        ticks,
        classes,
        rotation=60,
        ha="right",
        fontsize=8,
    )
    plt.yticks(
        ticks,
        classes,
        fontsize=8,
    )

    threshold = matrix.max() * 0.60

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value:
                plt.text(
                    j,
                    i,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > threshold else "black",
                )

    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "12_challenge_confusion_matrix.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    # Challenge metrics graph.
    metrics = pd.DataFrame(report_dict).T.loc[
        [c for c in classes if c in report_dict],
        ["precision", "recall", "f1-score"],
    ]

    metrics.plot(
        kind="bar",
        figsize=(17, 8),
    )
    plt.title(
        "Independent Synthetic Challenge — Precision / Recall / F1"
    )
    plt.xlabel("Intent")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=60, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "13_challenge_precision_recall_f1.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Challenge confidence distribution.
    plt.figure(figsize=(10, 6))
    plt.hist(
        confidence,
        bins=30,
        edgecolor="black",
    )
    plt.title(
        "Independent Synthetic Challenge — Confidence Distribution"
    )
    plt.xlabel("Maximum Predicted Probability")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "14_challenge_confidence_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

        # Challenge per-intent accuracy
    challenge_accuracy_by_class = (
        metrics["recall"]
        .sort_values()
    )

    plt.figure(figsize=(14, 8))

    challenge_accuracy_by_class.plot(
        kind="barh"
    )

    plt.title(
        "Independent Synthetic Challenge — Per-Intent Accuracy"
    )
    plt.xlabel("Recall / Class Accuracy")
    plt.ylabel("Intent")
    plt.xlim(0, 1.05)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "15_challenge_per_intent_accuracy.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

        # Challenge correct vs incorrect
    challenge_correct_count = int(correct.sum())
    challenge_wrong_count = int((~correct).sum())

    plt.figure(figsize=(7, 6))

    plt.bar(
        ["Correct", "Incorrect"],
        [
            challenge_correct_count,
            challenge_wrong_count,
        ],
    )

    plt.title(
        "Independent Synthetic Challenge — Correct vs Incorrect"
    )
    plt.ylabel("Number of Challenge Samples")
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "16_challenge_correct_vs_incorrect.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "samples": int(len(challenge)),
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "macro_precision": float(
            report_dict["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report_dict["macro avg"]["recall"]
        ),
        "macro_f1": float(
            report_dict["macro avg"]["f1-score"]
        ),
        "weighted_precision": float(
            report_dict["weighted avg"]["precision"]
        ),
        "weighted_recall": float(
            report_dict["weighted avg"]["recall"]
        ),
        "weighted_f1": float(
            report_dict["weighted avg"]["f1-score"]
        ),
        "correct": int(correct.sum()),
        "incorrect": int((~correct).sum()),
        "high_confidence_errors": int(
            len(high_conf_errors)
        ),

        "per_intent_accuracy": {
            str(c): float(report_dict[c]["recall"])
            for c in classes
        },

        "prediction_time_seconds": round(pred_time, 3),
        "confidence": confidence_stats,
    }


# ============================================================
# START
# ============================================================

pipeline_start = time.time()
run_timestamp = datetime.now().isoformat(timespec="seconds")

log("=" * 76)
log("NEUROCALC — FINAL 1M ML TRAINING PIPELINE")
log("=" * 76)
log(f"Python:   {platform.python_version()}")
log(f"Platform: {platform.platform()}")
log(f"Started:  {run_timestamp}")
log("=" * 76)


# ============================================================
# 1. DATASET
# ============================================================

DATASET = find_dataset()

dataset_size_mb = DATASET.stat().st_size / (1024 * 1024)

log(f"Dataset selected: {DATASET}")
log(f"Dataset size:     {dataset_size_mb:.2f} MB")
log("Loading dataset...")

df = pd.read_csv(DATASET)

required_columns = {"text", "intent"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset missing required columns: {sorted(missing_columns)}"
    )

original_rows = len(df)


# ============================================================
# 2. CLEANING
# ============================================================

log("Cleaning dataset...")

df = df[["text", "intent"]].copy()

df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

before_cleanup = len(df)

df = df[
    (df["text"] != "")
    & (df["intent"] != "")
].dropna()

after_cleanup = len(df)

duplicate_count = int(
    df.duplicated(subset=["text"]).sum()
)

# Keep first occurrence if duplicates exist.
df = (
    df
    .drop_duplicates(subset=["text"])
    .reset_index(drop=True)
)

final_rows = len(df)

if final_rows < 500:
    raise ValueError(
        f"Only {final_rows} usable rows remain. Dataset is too small."
    )

log(f"Original rows:          {original_rows:,}")
log(f"After empty/NA cleanup: {after_cleanup:,}")
log(f"Duplicate text rows:    {duplicate_count:,}")
log(f"Final usable rows:      {final_rows:,}")


# ============================================================
# 3. DATASET STATS
# ============================================================

intent_counts = (
    df["intent"]
    .value_counts()
    .sort_index()
)

dataset_stats = {
    "dataset_file": str(DATASET),
    "dataset_size_mb": round(dataset_size_mb, 2),
    "original_rows": int(original_rows),
    "rows_after_cleanup": int(after_cleanup),
    "duplicate_text_rows_removed": int(duplicate_count),
    "usable_rows": int(final_rows),
    "number_of_intents": int(df["intent"].nunique()),
    "min_class_count": int(intent_counts.min()),
    "max_class_count": int(intent_counts.max()),
    "class_balance_ratio": float(
        intent_counts.min() / intent_counts.max()
    ),
    "intent_counts": {
        str(k): int(v)
        for k, v in intent_counts.items()
    },
}

save_json(
    DATA_RESULTS_DIR / "dataset_statistics.json",
    dataset_stats,
)

intent_counts.rename("count").to_csv(
    DATA_RESULTS_DIR / "intent_distribution.csv"
)


# ============================================================
# 4. GRAPH — DATASET DISTRIBUTION
# ============================================================

plt.figure(figsize=(14, 8))
intent_counts.sort_values().plot(kind="barh")
plt.title("NeuroCalc Dataset Distribution by Intent")
plt.xlabel("Number of Samples")
plt.ylabel("Intent")
plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "01_dataset_intent_distribution.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

log("Creating stratified 80/20 train-test split...")

X = df["text"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

log(f"Training samples: {len(X_train):,}")
log(f"Testing samples:  {len(X_test):,}")

train_df = pd.DataFrame({
    "text": X_train.reset_index(drop=True),
    "intent": y_train.reset_index(drop=True),
})

test_df = pd.DataFrame({
    "text": X_test.reset_index(drop=True),
    "intent": y_test.reset_index(drop=True),
})

train_path = DATA_RESULTS_DIR / "train_data.csv"
test_path = DATA_RESULTS_DIR / "test_data.csv"

log(f"Saving training data: {train_path}")
train_df.to_csv(train_path, index=False)

log(f"Saving testing data:  {test_path}")
test_df.to_csv(test_path, index=False)

split_stats = {
    "total_samples": int(len(df)),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "training_percentage": round(
        100 * len(X_train) / len(df), 2
    ),
    "testing_percentage": round(
        100 * len(X_test) / len(df), 2
    ),
    "stratified": True,
    "random_state": RANDOM_STATE,
}

save_json(
    DATA_RESULTS_DIR / "split_statistics.json",
    split_stats,
)


# ============================================================
# 6. GRAPH — TRAIN/TEST DISTRIBUTION
# ============================================================

train_counts = y_train.value_counts().sort_index()
test_counts = y_test.value_counts().sort_index()

split_plot = pd.DataFrame({
    "Train": train_counts,
    "Test": test_counts,
})

split_plot.plot(
    kind="bar",
    figsize=(16, 8),
)

plt.title("Training vs Testing Samples by Intent")
plt.xlabel("Intent")
plt.ylabel("Number of Samples")
plt.xticks(rotation=60, ha="right")
plt.legend(title="Dataset Split")
plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "02_train_test_distribution.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 7. MODEL
# ============================================================

log("Building math-aware word + character TF-IDF model...")

word_vectorizer = TfidfVectorizer(
    preprocessor=add_math_tokens,
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"(?u)\b\w+\b",
    ngram_range=WORD_NGRAM_RANGE,
    min_df=WORD_MIN_DF,
    max_df=WORD_MAX_DF,
    max_features=WORD_MAX_FEATURES,
    sublinear_tf=True,
    dtype=np.float32,
)

char_vectorizer = TfidfVectorizer(
    preprocessor=add_math_tokens,
    analyzer="char_wb",
    ngram_range=CHAR_NGRAM_RANGE,
    min_df=CHAR_MIN_DF,
    max_df=CHAR_MAX_DF,
    max_features=CHAR_MAX_FEATURES,
    sublinear_tf=True,
    dtype=np.float32,
)

feature_union = FeatureUnion([
    ("word_tfidf", word_vectorizer),
    ("char_tfidf", char_vectorizer),
])

model = Pipeline([
    ("features", feature_union),
    (
        "classifier",
        LogisticRegression(
            solver=LOGREG_SOLVER,
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            class_weight=None,
            random_state=RANDOM_STATE,
        ),
    ),
])


# ============================================================
# 8. TRAIN
# ============================================================

log("Training model...")
training_start = time.time()

model.fit(X_train, y_train)

training_time_seconds = time.time() - training_start

log(
    f"Training finished in "
    f"{training_time_seconds / 60:.2f} minutes."
)


# ============================================================
# 9. MODEL METADATA
# ============================================================

features = model.named_steps["features"]
classifier = model.named_steps["classifier"]

word_fitted = features.transformer_list[0][1]
char_fitted = features.transformer_list[1][1]

word_features = word_fitted.get_feature_names_out()
char_features = char_fitted.get_feature_names_out()

feature_names = np.concatenate([
    np.array([f"WORD::{x}" for x in word_features]),
    np.array([f"CHAR::{x}" for x in char_features]),
])

actual_feature_count = len(feature_names)

model_path = MODEL_DIR / "neurocalc_intent_model_1m.joblib"

log(f"Saving trained model: {model_path}")
joblib.dump(model, model_path, compress=3)

model_stats = {
    "model_type": "Math-aware Word + Character TF-IDF + Logistic Regression",
    "math_operator_tokenization": True,
    "word_ngram_range": list(WORD_NGRAM_RANGE),
    "word_min_df": WORD_MIN_DF,
    "word_max_df": WORD_MAX_DF,
    "word_max_features": WORD_MAX_FEATURES,
    "word_actual_features": int(len(word_features)),
    "char_analyzer": "char_wb",
    "char_ngram_range": list(CHAR_NGRAM_RANGE),
    "char_min_df": CHAR_MIN_DF,
    "char_max_df": CHAR_MAX_DF,
    "char_max_features": CHAR_MAX_FEATURES,
    "char_actual_features": int(len(char_features)),
    "actual_combined_features": int(actual_feature_count),
    "logistic_regression_solver": LOGREG_SOLVER,
    "logistic_regression_C": LOGREG_C,
    "logistic_regression_max_iter": LOGREG_MAX_ITER,
    "class_weight": None,
    "classes": [str(x) for x in classifier.classes_],
    "random_state": RANDOM_STATE,
}

save_json(
    REPORT_DIR / "model_configuration.json",
    model_stats,
)

save_text(
    REPORT_DIR / "tfidf_feature_names.txt",
    "\n".join(feature_names),
)


# ============================================================
# 10. PREDICTION / CONFIDENCE
# ============================================================

log("Generating predictions...")

prediction_start = time.time()

pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)

prediction_time_seconds = time.time() - prediction_start

confidence = probabilities.max(axis=1)

accuracy = accuracy_score(y_test, pred)

correct = (
    y_test.reset_index(drop=True).to_numpy()
    == pred
)

top_k_indices = np.argsort(
    probabilities,
    axis=1
)[:, -TOP3_K:]

classes = classifier.classes_

top3_correct = np.array([
    actual in classes[row]
    for actual, row in zip(
        y_test.reset_index(drop=True).to_numpy(),
        top_k_indices,
    )
])

top3_accuracy = float(top3_correct.mean())

log(f"Accuracy:      {accuracy:.4%}")
log(f"Top-{TOP3_K} accuracy: {top3_accuracy:.4%}")
log(
    f"Prediction time: "
    f"{prediction_time_seconds:.2f} seconds."
)

# Cross-entropy / log loss.
try:
    multiclass_log_loss = float(
        log_loss(
            y_test,
            probabilities,
            labels=classes,
        )
    )
except Exception:
    multiclass_log_loss = None


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_test,
    pred,
    zero_division=0,
    output_dict=True,
)

report_text = classification_report(
    y_test,
    pred,
    zero_division=0,
)

save_text(
    REPORT_DIR / "classification_report.txt",
    report_text,
)

report_df = pd.DataFrame(report_dict).T

report_df.to_csv(
    REPORT_DIR / "classification_report.csv"
)


# ============================================================
# 12. TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "text": X_test.reset_index(drop=True),
    "actual_intent": y_test.reset_index(drop=True),
    "predicted_intent": pred,
    "confidence": confidence,
    "top3_correct": top3_correct,
    "correct": correct,
})

prediction_df.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False,
)

errors_df = prediction_df[
    ~prediction_df["correct"]
].copy()

errors_df.to_csv(
    REPORT_DIR / "misclassified_examples.csv",
    index=False,
)

high_confidence_errors = errors_df[
    errors_df["confidence"] >= 0.90
].copy()

high_confidence_errors.to_csv(
    REPORT_DIR / "high_confidence_errors.csv",
    index=False,
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    pred,
    labels=classes,
)

plt.figure(figsize=(17, 15))
plt.imshow(cm, interpolation="nearest")
plt.title("NeuroCalc Confusion Matrix")
plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.colorbar()

ticks = np.arange(len(classes))

plt.xticks(
    ticks,
    classes,
    rotation=60,
    ha="right",
    fontsize=8,
)

plt.yticks(
    ticks,
    classes,
    fontsize=8,
)

# Annotate reasonably large cells.
threshold = cm.max() * 0.60

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=6,
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
        )

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "03_confusion_matrix.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close()

pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
).to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)


# ============================================================
# 14. PRECISION / RECALL / F1
# ============================================================

metric_df = report_df.loc[
    [c for c in classes if c in report_df.index],
    ["precision", "recall", "f1-score"],
]

metric_df.plot(
    kind="bar",
    figsize=(17, 8),
)

plt.title("Precision, Recall and F1-Score by Intent")
plt.xlabel("Intent")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=60, ha="right")
plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "04_precision_recall_f1.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 15. PER-INTENT PERFORMANCE
# ============================================================

accuracy_by_class = (
    metric_df["recall"]
    .sort_values()
)

plt.figure(figsize=(14, 8))

accuracy_by_class.plot(
    kind="barh"
)

plt.title("Per-Intent Classification Performance")
plt.xlabel("Recall / Class Accuracy")
plt.ylabel("Intent")
plt.xlim(0, 1.05)
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "05_per_intent_performance.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 16. CONFIDENCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    confidence,
    bins=30,
    edgecolor="black",
)

plt.title("Prediction Confidence Distribution")
plt.xlabel("Maximum Predicted Probability")
plt.ylabel("Number of Test Samples")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "06_confidence_distribution.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 17. CORRECT VS INCORRECT
# ============================================================

correct_count = int(correct.sum())
wrong_count = int((~correct).sum())

plt.figure(figsize=(7, 6))

plt.bar(
    ["Correct", "Incorrect"],
    [correct_count, wrong_count],
)

plt.title("Correct vs Incorrect Predictions")
plt.ylabel("Number of Test Samples")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "07_correct_vs_incorrect.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 18. TOP FEATURES
# ============================================================

log("Extracting important model features...")

coef = classifier.coef_

top_feature_rows = []

for class_index, class_name in enumerate(
    classifier.classes_
):
    weights = coef[class_index]

    top_indices = np.argsort(
        weights
    )[-TOP_FEATURES_PER_CLASS:][::-1]

    for rank, feature_index in enumerate(
        top_indices,
        start=1,
    ):
        top_feature_rows.append({
            "intent": class_name,
            "rank": rank,
            "feature": feature_names[feature_index],
            "weight": float(
                weights[feature_index]
            ),
        })

top_features_df = pd.DataFrame(
    top_feature_rows
)

top_features_df.to_csv(
    REPORT_DIR / "top_features_by_intent.csv",
    index=False,
)


# ============================================================
# 19. TOP FEATURE GRAPH
# ============================================================

feature_plot_df = top_features_df[
    top_features_df["rank"] <= 5
].copy()

plt.figure(figsize=(17, 13))

y_offset = 0

for class_name in classes:

    subset = (
        feature_plot_df[
            feature_plot_df["intent"] == class_name
        ]
        .sort_values("weight")
    )

    if subset.empty:
        continue

    positions = (
        np.arange(len(subset))
        + y_offset
    )

    plt.barh(
        positions,
        subset["weight"],
        height=0.7,
        label=class_name,
    )

    y_offset += len(subset) + 1

plt.title("Top TF-IDF Features by Intent")
plt.xlabel("Logistic Regression Weight")
plt.ylabel("Intent Groups")
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=7,
)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "08_top_features_by_intent.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()



# ============================================================
# INDEPENDENT CHALLENGE EVALUATION
# ============================================================

challenge_results = evaluate_independent_challenge(
    model,
    CHALLENGE_DATASET,
    REPORT_DIR,
)

# High-confidence validation errors (confidence >= 90%)
high_conf_count = int(
    len(high_confidence_errors)
)

comparison = pd.DataFrame([
    {
        "evaluation": "Synthetic validation",
        "samples": int(len(X_test)),
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_accuracy),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "incorrect": int(wrong_count),
        "high_confidence_errors": int(high_conf_count),
    },
    {
        "evaluation": "Independent synthetic challenge",
        "samples": challenge_results["samples"],
        "accuracy": challenge_results["accuracy"],
        "top3_accuracy": challenge_results["top3_accuracy"],
        "macro_f1": challenge_results["macro_f1"],
        "weighted_f1": challenge_results["weighted_f1"],
        "incorrect": challenge_results["incorrect"],
        "high_confidence_errors": challenge_results["high_confidence_errors"],
    },
])

comparison.to_csv(
    REPORT_DIR / "validation_vs_challenge.csv",
    index=False,
)

# ============================================================
# VALIDATION VS CHALLENGE — PER-INTENT COMPARISON
# ============================================================

validation_per_intent = {
    str(c): float(report_dict[c]["recall"])
    for c in classes
}

challenge_per_intent = challenge_results[
    "per_intent_accuracy"
]

per_intent_comparison = pd.DataFrame({
    "Validation": [
        validation_per_intent[c]
        for c in classes
    ],
    "Challenge": [
        challenge_per_intent.get(c, 0.0)
        for c in classes
    ],
}, index=classes)

plt.figure(figsize=(17, 9))

per_intent_comparison.plot(
    kind="bar",
    figsize=(17, 9),
)

plt.title(
    "Validation vs Independent Challenge — Per-Intent Accuracy"
)
plt.xlabel("Intent")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=60, ha="right")
plt.legend(title="Evaluation Set")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "17_validation_vs_challenge_per_intent.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close()

per_intent_comparison.to_csv(
    REPORT_DIR / "validation_vs_challenge_per_intent.csv"
)

comparison.set_index(
    "evaluation"
)[["accuracy", "macro_f1", "weighted_f1"]].plot(
    kind="bar",
    figsize=(11, 7),
)

plt.title(
    "Synthetic Validation vs Independent Challenge"
)

plt.xlabel("Evaluation Set")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "11_validation_vs_challenge.png",
    dpi=220,
    bbox_inches="tight",
)

plt.close()

# Persist a compact machine-readable comparison.
save_json(
    REPORT_DIR / "validation_challenge_summary.json",
    {
        "validation": {
            "accuracy": float(accuracy),
            "macro_f1": float(report_dict["macro avg"]["f1-score"]),
            "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        },
        "challenge": challenge_results,
        "challenge_used_for_training": False,
        "challenge_exact_overlap_with_main_dataset": 0,
    },
)

# ============================================================
# 20. TOP-3 GRAPH
# ============================================================

plt.figure(figsize=(7, 6))

plt.bar(
    ["Top-1", "Top-3"],
    [accuracy, top3_accuracy],
)

plt.ylim(0, 1.05)
plt.title("Top-1 vs Top-3 Classification Accuracy")
plt.ylabel("Accuracy")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "09_top3_accuracy.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 21. HIGH-CONFIDENCE ERROR GRAPH
# ============================================================

high_conf_count = int(
    len(high_confidence_errors)
)

low_conf_correct_count = int(
    correct.sum()
)

plt.figure(figsize=(8, 6))

plt.bar(
    ["All Correct", "High-Confidence Errors"],
    [low_conf_correct_count, high_conf_count],
)

plt.title("Correct Predictions vs High-Confidence Errors")
plt.ylabel("Number of Test Samples")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "10_high_confidence_errors.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 22. MISCLASSIFICATION ANALYSIS
# ============================================================

misclassification_summary = (
    prediction_df[
        ~prediction_df["correct"]
    ]
    .groupby(
        [
            "actual_intent",
            "predicted_intent",
        ]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False,
    )
)

misclassification_summary.to_csv(
    REPORT_DIR / "misclassification_summary.csv",
    index=False,
)


# ============================================================
# 23. CONFIDENCE / CALIBRATION
# ============================================================

correct_mean_confidence = (
    float(
        prediction_df.loc[
            prediction_df["correct"],
            "confidence",
        ].mean()
    )
)

incorrect_mean_confidence = (
    float(
        prediction_df.loc[
            ~prediction_df["correct"],
            "confidence",
        ].mean()
    )
)

confidence_stats = {
    "mean_confidence": float(
        np.mean(confidence)
    ),
    "median_confidence": float(
        np.median(confidence)
    ),
    "min_confidence": float(
        np.min(confidence)
    ),
    "max_confidence": float(
        np.max(confidence)
    ),
    "correct_mean_confidence": correct_mean_confidence,
    "incorrect_mean_confidence": incorrect_mean_confidence,
    "samples_below_0_50": int(
        np.sum(confidence < 0.50)
    ),
    "samples_below_0_70": int(
        np.sum(confidence < 0.70)
    ),
    "samples_above_or_equal_0_90": int(
        np.sum(confidence >= 0.90)
    ),
    "high_confidence_errors": high_conf_count,
    "expected_calibration_error": calculate_ece(
        confidence,
        correct,
        bins=10,
    ),
    "multiclass_log_loss": multiclass_log_loss,
}

save_json(
    REPORT_DIR / "confidence_statistics.json",
    confidence_stats,
)


# ============================================================
# 24. SYSTEM + RUN CONFIG
# ============================================================

run_config = {
    "timestamp": run_timestamp,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "dataset": str(DATASET),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "operator_aware_features": True,
    "feature_spaces": [
        "word_tfidf",
        "char_wb_tfidf",
    ],
}

save_json(
    REPORT_DIR / "run_configuration.json",
    run_config,
)


# ============================================================
# 25. OVERALL SUMMARY
# ============================================================

total_time = time.time() - pipeline_start

summary = {
    "project": "NeuroCalc",
    "experiment": (
        "1M dataset — math-aware word+character "
        "intent classification"
    ),
    "timestamp": run_timestamp,

    "dataset": dataset_stats,

    "split": split_stats,

    "model": model_stats,

    "performance": {
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_accuracy),

        "macro_precision": float(
            report_dict["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report_dict["macro avg"]["recall"]
        ),
        "macro_f1": float(
            report_dict["macro avg"]["f1-score"]
        ),

        "weighted_precision": float(
            report_dict["weighted avg"]["precision"]
        ),
        "weighted_recall": float(
            report_dict["weighted avg"]["recall"]
        ),
        "weighted_f1": float(
            report_dict["weighted avg"]["f1-score"]
        ),

        "correct_predictions": correct_count,
        "incorrect_predictions": wrong_count,
        "high_confidence_errors": high_conf_count,

        "training_time_seconds": round(
            training_time_seconds,
            2,
        ),
        "prediction_time_seconds": round(
            prediction_time_seconds,
            2,
        ),
        "total_pipeline_time_seconds": round(
            total_time,
            2,
        ),
    },

    "outputs": {
        "model": str(model_path),
        "graphs": str(GRAPH_DIR),
        "report": str(REPORT_DIR),
        "train_data": str(train_path),
        "test_data": str(test_path),
    },
}

save_json(
    REPORT_DIR / "training_summary.json",
    summary,
)


# ============================================================
# 26. HUMAN-READABLE REPORT
# ============================================================

report_lines = [
    "NEUROCALC — FINAL ML TRAINING EXPERIMENT REPORT",
    "=" * 76,
    "",
    f"Experiment date: {run_timestamp}",
    f"Dataset: {DATASET.name}",
    f"Original rows: {original_rows:,}",
    f"Usable rows: {final_rows:,}",
    f"Duplicate texts removed: {duplicate_count:,}",
    f"Intents: {df['intent'].nunique()}",
    f"Training rows: {len(X_train):,}",
    f"Testing rows: {len(X_test):,}",
    "",
    "FEATURE ENGINEERING",
    "-" * 76,
    "Math-aware operator tokenization: ENABLED",
    "Word TF-IDF: 1–3 grams",
    f"Word maximum features: {WORD_MAX_FEATURES:,}",
    "Character TF-IDF: char_wb 3–5 grams",
    f"Character maximum features: {CHAR_MAX_FEATURES:,}",
    f"Combined actual features: {actual_feature_count:,}",
    "",
    "CLASSIFIER",
    "-" * 76,
    "Logistic Regression",
    f"Solver: {LOGREG_SOLVER}",
    f"C: {LOGREG_C}",
    f"Maximum iterations: {LOGREG_MAX_ITER}",
    "",
    "RESULTS",
    "-" * 76,
    f"Accuracy: {accuracy:.4%}",
    f"Top-{TOP3_K} accuracy: {top3_accuracy:.4%}",
    f"Macro Precision: {report_dict['macro avg']['precision']:.4%}",
    f"Macro Recall: {report_dict['macro avg']['recall']:.4%}",
    f"Macro F1: {report_dict['macro avg']['f1-score']:.4%}",
    f"Weighted Precision: {report_dict['weighted avg']['precision']:.4%}",
    f"Weighted Recall: {report_dict['weighted avg']['recall']:.4%}",
    f"Weighted F1: {report_dict['weighted avg']['f1-score']:.4%}",
    f"Correct predictions: {correct_count:,}",
    f"Incorrect predictions: {wrong_count:,}",
    f"High-confidence errors (>=90%): {high_conf_count:,}",
    f"Mean confidence: {confidence_stats['mean_confidence']:.4%}",
    f"Expected calibration error: {confidence_stats['expected_calibration_error']:.4%}",
    "",
    "INDEPENDENT SYNTHETIC CHALLENGE",
    "-" * 76,
    f"Challenge file: {CHALLENGE_DATASET.name}",
    f"Challenge samples: {challenge_results['samples']:,}",
    "Used during training: NO",
    "Exact overlap with main dataset: 0",
    f"Challenge Accuracy: {challenge_results['accuracy']:.4%}",
    f"Challenge Top-3 Accuracy: {challenge_results['top3_accuracy']:.4%}",
    f"Challenge Macro F1: {challenge_results['macro_f1']:.4%}",
    f"Challenge Weighted F1: {challenge_results['weighted_f1']:.4%}",
    f"Challenge Correct: {challenge_results['correct']:,}",
    f"Challenge Incorrect: {challenge_results['incorrect']:,}",
    f"Challenge High-confidence errors: {challenge_results['high_confidence_errors']:,}",
    "",
    "TIMING",
    "-" * 76,
    f"Training time: {training_time_seconds / 60:.2f} minutes",
    f"Prediction time: {prediction_time_seconds:.2f} seconds",
    f"Total pipeline: {total_time / 60:.2f} minutes",
    "",
    "OUTPUTS",
    "-" * 76,
    f"Model: {model_path}",
    f"Graphs: {GRAPH_DIR}",
    f"Report: {REPORT_DIR}",
    f"Train data: {train_path}",
    f"Test data: {test_path}",
    "",
    "LIMITATION",
    "-" * 76,
    "The dataset is synthetic. Validation performance on synthetic examples",
    "does not by itself establish real-world natural-language accuracy.",
    "A separate human-written evaluation set should be used for final claims.",
]

save_text(
    REPORT_DIR / "experiment_report.txt",
    "\n".join(report_lines),
)


# ============================================================
# 27. MODEL CARD
# ============================================================

model_card = f"""# NeuroCalc ML Model Card

## Model
Math-aware Word + Character TF-IDF + Logistic Regression.

## Intended use
Classify natural-language calculator requests into NeuroCalc intents.
The model should identify *what operation the user wants*. It should not
be treated as the arithmetic engine.

## Dataset
File: `{DATASET.name}`

Rows used: {final_rows:,}
Intents: {df['intent'].nunique()}
Train: {len(X_train):,}
Test: {len(X_test):,}

## Feature engineering
Math operators are converted to explicit tokens such as:
- OP_ADD
- OP_SUB
- OP_MUL
- OP_DIV
- OP_POWER
- OP_PERCENT
- OP_EQUAL

The model then learns from both word and character TF-IDF features.

## Performance
Accuracy: {accuracy:.4%}
Independent challenge accuracy: {challenge_results["accuracy"]:.4%}
Top-3 accuracy: {top3_accuracy:.4%}
Macro F1: {report_dict['macro avg']['f1-score']:.4%}
Weighted F1: {report_dict['weighted avg']['f1-score']:.4%}

## Safety behavior
NeuroCalc should use the model for intent understanding and deterministic
math libraries for actual calculations. A low-confidence prediction should
not silently become an unrelated calculation.

## Limitations
The dataset is synthetic and generated for development/education.
Performance on this dataset should not be reported as real-world accuracy.
Use a separate human-written evaluation set for final claims.
"""

save_text(
    REPORT_DIR / "model_card.md",
    model_card,
)


# ============================================================
# 28. FINAL CONSOLE
# ============================================================

log("")
log("=" * 76)
log("TRAINING + INDEPENDENT EVALUATION COMPLETE")
log("=" * 76)
log(f"Validation accuracy:    {accuracy:.4%}")
log(f"Challenge accuracy:     {challenge_results['accuracy']:.4%}")
log(f"Validation Top-{TOP3_K} accuracy: {top3_accuracy:.4%}")
log(f"Macro F1:          {report_dict['macro avg']['f1-score']:.4%}")
log(f"Weighted F1:       {report_dict['weighted avg']['f1-score']:.4%}")
log(f"Correct:           {correct_count:,}")
log(f"Incorrect:         {wrong_count:,}")
log(f"High-conf errors:  {high_conf_count:,}")
log(f"Training time:     {training_time_seconds / 60:.2f} minutes")
log(f"Total pipeline:    {total_time / 60:.2f} minutes")
log("")
log("MODEL:")
log(f"  {model_path}")
log("")
log("REPORT DIRECTORY:")
log(f"  {RESULTS_DIR}")
log("")
log("GRAPHS:")
for graph_file in sorted(GRAPH_DIR.glob("*.png")):
    log(f"  {graph_file.name}")
log("")
log("IMPORTANT:")
log("This model is for intent understanding; exact math stays deterministic.")
log("Use experiment_report.txt + training_summary.json in your report.")
log("=" * 76)
