
# 🎓 Student Performance Classification
Predicting student academic performance (**Low / Medium / High**) using behavioral engagement data from an e-learning platform, built with a full ML pipeline — EDA, preprocessing, model comparison, and hyperparameter tuning.

![Model Accuracy Comparison](model_accuracy_comparison.png)

## 📌 Overview

Educational institutions increasingly rely on data-driven insights to understand what drives student outcomes. This project analyzes academic activity and engagement data from the **xAPI-Edu-Data** dataset to build a classification model that predicts a student's performance level.

- **Dataset**: 480 student records, 13 original attributes (10 used after feature selection)
- **Target**: `Class` — Low (0–69), Medium (70–89), High (90–100)
- **Task**: Multi-class classification

## 🗂 Dataset Description

| Feature | Description |
|---|---|
| Gender | Male / Female |
| StageID | lowerlevel / MiddleSchool / HighSchool |
| SectionID | Classroom section (A/B/C) |
| Relation | Parent responsible for the student (Mum/Father) |
| raisedhands | Times student raised hand in class (0–100) |
| VisITedResources | Times student visited course content (0–100) |
| AnnouncementsView | Times student checked new announcements (0–100) |
| Discussion | Times student participated in discussion groups (0–100) |
| ParentschoolSatisfaction | Parent satisfaction level (Good/Bad) |
| StudentAbsenceDays | Above-7 / Under-7 days absent |
| **Class** | **Target** — L / M / H performance level |

## 🔍 Approach

1. **EDA** — Histograms, pairplots, and categorical breakdowns to explore how engagement and absence relate to performance.
2. **Feature Selection** — Dropped `Institute`, `SectionID`, `Semester`, and `Topic` (low signal / high cardinality / domain judgment).
3. **Preprocessing** — Label encoding for categorical features, ordinal mapping for the target (L=0, M=1, H=2), and `StandardScaler` normalization.
4. **Modeling** — Trained and compared:
   - Logistic Regression
   - Decision Tree (baseline → pruned to address overfitting)
   - Random Forest (default → tuned via `GridSearchCV`)
5. **Evaluation** — Confusion matrices, precision/recall/F1, and accuracy on a held-out test set.

## 📊 Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression | 75.0% |
| Decision Tree (baseline) | 72.9% |
| Decision Tree (tuned) | 76.0% |
| Random Forest | 81.25% |
| **Random Forest (GridSearchCV-tuned)** | **83.3%** |

**Key finding**: `StudentAbsenceDays` and platform engagement metrics (`VisITedResources`, `raisedhands`) were the strongest predictors of performance — stronger than demographic factors like gender or parental relation.

**Best hyperparameters** (Random Forest via GridSearchCV):
```
max_depth: 8
min_samples_leaf: 1
n_estimators: 1000
```

## 🛠 Tech Stack

- Python (pandas, numpy)
- scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV)
- seaborn, matplotlib (EDA & visualization)

## 🚀 Getting Started

```bash
git clone <your-repo-url>
cd student-performance-classification
pip install -r requirements.txt
jupyter notebook Student_Performance_Classification.ipynb
```

### requirements.txt
```
pandas
numpy
scikit-learn
seaborn
matplotlib
```

## 📁 Project Structure

```
.
├── xAPI-Edu-Data.csv
├── Student_Performance_Classification.ipynb
├── model_accuracy_comparison.png
└── README.md
```

## 📄 Dataset Source

[xAPI-Edu-Data (Kaggle)](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data)

## 📃 License

This project is open source and available under the [MIT License](LICENSE).
