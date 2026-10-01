# NeuroCalc ML Model Card

## Model
Math-aware Word + Character TF-IDF + Logistic Regression.

## Intended use
Classify natural-language calculator requests into NeuroCalc intents.
The model should identify *what operation the user wants*. It should not
be treated as the arithmetic engine.

## Dataset
File: `neurocalc_dataset_1m.csv`

Rows used: 1,000,000
Intents: 20
Train: 800,000
Test: 200,000

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
Accuracy: 100.0000%
Independent challenge accuracy: 94.1000%
Top-3 accuracy: 100.0000%
Macro F1: 100.0000%
Weighted F1: 100.0000%

## Safety behavior
NeuroCalc should use the model for intent understanding and deterministic
math libraries for actual calculations. A low-confidence prediction should
not silently become an unrelated calculation.

## Limitations
The dataset is synthetic and generated for development/education.
Performance on this dataset should not be reported as real-world accuracy.
Use a separate human-written evaluation set for final claims.
