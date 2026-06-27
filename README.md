# 🏥 Insurance Charges Prediction

Predicting medical insurance charges using regression models and interaction terms, based on the US Medical Insurance dataset.

---

## 📌 Project Overview

This project explores what drives medical insurance premiums and builds a predictive model using machine learning. The workflow covers EDA, feature engineering (interaction terms), and comparison of multiple regression models.

**Target variable:** `charges` (annual insurance premium in USD)

---

## 📁 Dataset

- **Source:** [Kaggle - US Medical Insurance Costs](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Size:** 1,338 rows × 7 columns
- **No missing values**

| Column | Type | Description |
|---|---|---|
| `age` | int | Age of the policyholder |
| `sex` | str | Gender (male/female) |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of dependents |
| `smoker` | str | Smoking status (yes/no) |
| `region` | str | US region (4 categories) |
| `charges` | float | Annual insurance charge (target) |

---

## 🔍 Exploratory Data Analysis

### Distribution of Charges
- **Mean:** $13,270 | **Median:** $9,382
- **Skewness:** 1.52 (right-skewed) → majority pay below average, but a small segment pays extremely high
- **Kurtosis:** 1.61 → heavy tails; a few high-cost outliers significantly inflate the mean

> Actuary note: High skewness + kurtosis = non-uniform risk distribution → standard premium pricing based on mean alone would underestimate tail risk.

### Key Correlations with `charges`

| Feature | Correlation |
|---|---|
| `smoker_yes` | **0.787** |
| `age` | 0.299 |
| `bmi` | 0.198 |
| Others | < 0.08 |

**Smoker status is by far the dominant predictor.**

---

## ⚙️ Preprocessing

- Categorical encoding via `pd.get_dummies()` with `drop_first=True` (dummy variable trap avoidance)
- Train/Test split: **75% / 25%**, `random_state=15`
- Feature scaling: `StandardScaler`

---

## 🤖 Models & Results

### Baseline Linear Models (no interaction terms)

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | 0.74 | 0.79 |
| Ridge | 0.74 | 0.79 |
| Lasso | 0.74 | 0.79 |
| ElasticNet | 0.65 | 0.71 |

All three linear models hit the same ceiling at R² = 0.79. ElasticNet underperforms, too much regularization killing signal.

---

### Linear Regression + Interaction Terms

Added two engineered features:
- `bmi_smoker` = `bmi × smoker_yes`
- `age_smoker` = `age × smoker_yes`

| | Train R² | Test R² | CV R² (5-fold avg) |
|---|---|---|---|
| Linear + Interactions | 0.82 | **0.88** | 0.836 |

Interaction terms boosted R² from 0.79 → 0.88 , capturing that BMI and age have amplified effects *specifically for smokers*.

---

### Tree-Based Models

| Model | Max Depth | Train R² | Test R² |
|---|---|---|---|
| Decision Tree | 4 | 0.865 | 0.864 |
| Random Forest | 4 | 0.872 | **0.871** |

Random Forest at max_depth=4 is the sweet spot, negligible gap between train/test, no overfitting. Deeper trees (depth 7-8) showed clear overfitting (train ~0.93, test ~0.80).

### Feature Importance (Decision Tree)

| Feature | Importance |
|---|---|
| `smoker_yes` | 70.3% |
| `bmi` | 17.9% |
| `age` | 11.4% |
| Others | < 0.5% |

---

## 📊 Model Comparison Summary

| Model | Test R² | Notes |
|---|---|---|
| Linear / Ridge / Lasso | 0.79 | Plateau, linear assumption too strict |
| ElasticNet | 0.71 | Over-regularized |
| **Linear + Interactions** | **0.88** | Best interpretable model |
| Decision Tree (depth=4) | 0.86 | Clean, no overfitting |
| Random Forest (depth=4) | **0.87** | Best overall generalization |

---

## 🧠 Key Insights

1. **Smoking is the biggest risk factor** — 70%+ of predictive power comes from `smoker_yes` alone
2. **Non-smokers paying >$30K are mostly 50+ years old** (avg age: 56.2) → age catches up for non-smokers
3. **BMI × Smoking interaction matters** — high BMI is only high-risk *if* the person also smokes
4. **Linear models plateau at R²=0.79** — interaction terms or nonlinear models are required to go further
5. **Overfitting is depth-sensitive in trees** — depth > 6 starts degrading out-of-sample performance

---

## 🛠️ Tech Stack

- Python 3.14
- `pandas`, `numpy`, `scipy`
- `scikit-learn` (LinearRegression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest)
- `matplotlib`, `seaborn`

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/insurance-charges-prediction.git
cd insurance-charges-prediction
pip install -r requirements.txt
jupyter notebook analysis2.ipynb
```

**Dataset:** Download `insurance.csv` from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) and place it in the project root (or update the path in cell 1).

---
