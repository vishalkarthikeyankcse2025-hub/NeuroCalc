import re
import sys
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIG
# ============================================================

DATASET = "data/neurocalc_dataset_1m.csv"
OUTPUT_MODEL = "models/neurocalc_android.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# Word TF-IDF
WORD_NGRAM_RANGE = (1, 3)
WORD_MIN_DF = 2
WORD_MAX_DF = 0.995
WORD_MAX_FEATURES = 100_000


# Character TF-IDF
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_DF = 0.995
CHAR_MAX_FEATURES = 50_000


# Logistic Regression
LOGREG_SOLVER = "saga"
LOGREG_C = 4.0
LOGREG_MAX_ITER = 150


# ============================================================
# MATH-AWARE PREPROCESSING
# ============================================================

def add_math_tokens(text: str) -> str:

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
        r"(?<=[A-Za-z0-9)\]])\s*\^\s*(?=[A-Za-z0-9(\[])",
        " OP_POWER ",
        s,
    )

    s = s.replace("%", " OP_PERCENT ")
    s = s.replace("=", " OP_EQUAL ")
    s = s.replace("[", " OP_LBRACKET ")
    s = s.replace("]", " OP_RBRACKET ")

    s = s.replace("*", " OP_MUL ")
    s = s.replace("/", " OP_DIV ")

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


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NEUROCALC — ANDROID COMPATIBLE MODEL TRAINING")
print("=" * 70)

print("Python :", sys.version)
print("NumPy  :", np.__version__)

import sklearn
print("sklearn:", sklearn.__version__)

print("joblib :", joblib.__version__)

print()
print("Loading dataset...")
print(DATASET)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET)

df = df[["text", "intent"]].copy()

df["text"] = df["text"].astype(str).str.strip()
df["intent"] = df["intent"].astype(str).str.strip()

df = df[
    (df["text"] != "") &
    (df["intent"] != "")
]

df = df.dropna()

df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

print()
print("Final dataset rows:", len(df))
print("Intent classes:", df["intent"].nunique())

print()
print("Class distribution:")
print(df["intent"].value_counts().sort_index())


# ============================================================
# SPLIT
# ============================================================

X = df["text"]
y = df["intent"]

print()
print("Creating 80/20 stratified split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("Building NeuroCalc model...")

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


features = FeatureUnion([
    ("word_tfidf", word_vectorizer),
    ("char_tfidf", char_vectorizer),
])


model = Pipeline([
    ("features", features),

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
# TRAIN
# ============================================================

print()
print("Training...")
print("This may take several minutes.")

start = time.time()

model.fit(X_train, y_train)

elapsed = time.time() - start

print()
print(f"Training time: {elapsed / 60:.2f} minutes")


# ============================================================
# EVALUATION
# ============================================================

print()
print("Evaluating model...")

pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

print()
print("=" * 70)
print("ANDROID MODEL VALIDATION")
print("=" * 70)

print(f"Accuracy: {accuracy:.4%}")

print()
print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)


# ============================================================
# TEST QUERIES
# ============================================================

print()
print("=" * 70)
print("QUICK TEST")
print("=" * 70)

tests = [
    "25 + 10",
    "25 multiplied by 4",
    "square root of 144",
    "50 divided by 5",
    "what is 10 percent of 200",
]

for query in tests:

    prediction = model.predict([query])[0]

    probabilities = model.predict_proba([query])[0]

    confidence = float(
        probabilities.max()
    )

    print(
        f"{query} -> "
        f"{prediction} "
        f"(confidence={confidence:.4f})"
    )


# ============================================================
# SAVE
# ============================================================

print()
print("Saving Android-compatible model...")

import os
os.makedirs(
    os.path.dirname(OUTPUT_MODEL),
    exist_ok=True
)

joblib.dump(
    model,
    OUTPUT_MODEL,
    compress=3
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print("Saved:")
print(OUTPUT_MODEL)