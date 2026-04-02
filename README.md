# Electronic Device Rating Prediction Model

A machine learning classification project that predicts whether a laptop/electronic device will receive a **Good Rating** or **Bad Rating** based on its hardware specifications and pricing data.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

This project builds a binary classification pipeline to predict electronic device ratings from product specifications. The workflow covers the complete machine learning lifecycle: exploratory data analysis, preprocessing, feature engineering, model training, hyperparameter tuning, and serialization for deployment.

**Target Variable:** `rating` — `Good Rating` (0) or `Bad Rating` (1)

---

## Dataset

| Split | File | Samples |
|-------|------|---------|
| Training | `train.xlsx` | 579 |
| Test | `test.xlsx` | 144 |

Each record contains 18 attributes describing a laptop's hardware, software, and market reception.

| Column | Description |
|--------|-------------|
| `brand` | Laptop brand (e.g., HP, MSI) |
| `processor_brand` | CPU brand (Intel / AMD / Apple) |
| `processor_name` | CPU model (e.g., Core i5, Ryzen 5) |
| `processor_gnrtn` | CPU generation (4th – 12th) |
| `ram_gb` | RAM size (e.g., 8GB, 16GB) |
| `ram_type` | RAM type (DDR4, DDR5) |
| `ssd` | SSD storage capacity |
| `hdd` | HDD storage capacity |
| `os` | Operating system and bit version |
| `graphic_card_gb` | Dedicated GPU memory |
| `weight` | Weight category (Light / Heavy) |
| `warranty` | Warranty period |
| `Touchscreen` | Touchscreen availability (Yes / No) |
| `msoffice` | MS Office pre-installed (Yes / No) |
| `Price` | Device price (INR) |
| `Number of Ratings` | Total customer ratings |
| `Number of Reviews` | Total customer reviews |
| `rating` | **Target** — Good Rating / Bad Rating |

---

## Features

After feature selection, 9 features were retained for model training:

| Feature | Selection Reason |
|---------|-----------------|
| `ram_gb` | Strong correlation with rating |
| `ram_type` | Significant via Chi-squared test |
| `ssd` | Selected via ANOVA |
| `warranty` | Significant via Chi-squared test |
| `msoffice` | Significant via Chi-squared test |
| `Price` | Strong correlation with rating |
| `Number of Ratings` | Strong correlation with rating |
| `Number of Reviews` | Strong correlation with rating |
| `os-bits` | Derived from `os` column |

Features dropped: `processor_name`, `processor_gnrtn`, `processor_brand`, `os`, `brand`, `Touchscreen`, `hdd`, `graphic_card_gb`.

---

## Methodology

### Preprocessing

1. **Parsing** — Extracted `os-bits` from the `os` column; removed `GB` suffixes from storage/memory columns.
2. **Deduplication** — Removed duplicate rows.
3. **Encoding**
   - *Ordinal Encoding* (order-preserving): `rating`, `processor_gnrtn`, `warranty`, `processor_name`
   - *Label Encoding* (no natural order): `brand`, `processor_brand`, `ram_type`, `os`, `weight`, `Touchscreen`, `msoffice`
4. **Scaling** — MinMaxScaler (0–1 normalization) applied to `Price`, `Number of Ratings`, `Number of Reviews`.
5. **Outlier Removal** — Outliers detected and removed from numerical features.
6. **Class Imbalance** — SMOTE (Synthetic Minority Over-sampling Technique) applied to the training set to balance classes.

### Feature Selection

Three statistical methods were used to rank and select features:
- **Correlation Analysis** (numeric features vs. target)
- **Chi-Squared Test** (categorical features vs. target)
- **ANOVA F-test** (numeric features vs. categorical target)

---

## Models

Five classification algorithms were trained and evaluated:

| Model | Key Hyperparameters |
|-------|---------------------|
| Logistic Regression | `C=20`, `penalty='l1'`, `solver='liblinear'` |
| Random Forest | `max_depth=5`, `n_estimators=64` (GridSearchCV) |
| Decision Tree | `max_depth=3`, `min_samples_leaf=15`, `criterion='gini'` |
| XGBoost | `max_depth=10`, `n_estimators=100`, `min_child_weight=5` |
| AdaBoost | `n_estimators=50`, `learning_rate=0.6` (Decision Tree base) |

All trained models and preprocessing objects (encoders, scalers) are serialized as `.pkl` files for reproducible inference.

---

## Results

Models were evaluated on the held-out test set (144 samples) using accuracy, log loss, confusion matrix, and classification report (precision, recall, F1-score). **Random Forest** was selected as the primary model for test predictions.

---

## Project Structure

```
├── Classification_Electronic_Device_Rating_Prediction.ipynb   # Training pipeline
├── Test Script.ipynb                                           # Evaluation on test set
├── train.xlsx                                                  # Training data (579 samples)
├── test.xlsx                                                   # Test data (144 samples)
└── README.md
```

Serialized artifacts produced during training (not committed):

```
logisticRegression.pkl
randomForest.pkl
decisionTree.pkl
XGBoost.pkl
AdaBoost.pkl
<encoder/scaler>.pkl   # Preprocessing objects
```

---

## Usage

### Training

Open and run all cells in `Classification_Electronic_Device_Rating_Prediction.ipynb`. This will:
1. Load and preprocess `train.xlsx`
2. Train and tune all five models
3. Save pickled models and preprocessing objects to disk

### Inference / Testing

Open and run all cells in `Test Script.ipynb`. This will:
1. Load `test.xlsx` and the pickled preprocessing objects
2. Apply the identical preprocessing pipeline
3. Generate predictions from each model and compute accuracy

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, models, evaluation |
| `xgboost` | XGBoost classifier |
| `imbalanced-learn` | SMOTE for class balancing |
| `openpyxl` | Reading `.xlsx` files |
| `pickle` | Model serialization |

Install all dependencies via pip:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn openpyxl
```
