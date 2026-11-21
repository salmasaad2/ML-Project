# Heart Disease Prediction – DS471 Machine Learning Project

Prince Musaid Bin Abdulrahman University – College of Computing and Informatics

## Team
- Hala Rafat Almasri
- Ghala Faisal Almashari
- Salma Almaimoni
- Alanoud Ahmed Zain

## Problem
Binary classification: predict whether a patient has heart disease (1) or not (0)
from 13 clinical features.

## Dataset
UCI Heart Disease – Cleveland subset (303 patients, 13 features).
https://archive.ics.uci.edu/dataset/45/heart+disease
The notebook loads the data directly from UCI, so no download is needed.

## Workflow
1. Data cleaning, EDA and feature relevance
2. Stratified 70/15/15 train/validation/test split
3. Preprocessing (median/mode imputation, one-hot encoding, standard scaling) fitted on training data only
4. Models: Logistic Regression, Random Forest, Neural Network (Keras)
5. Hyperparameter tuning with 5-fold stratified cross-validation (ROC-AUC)
6. Final evaluation on the test set

## Results (test set, 46 patients)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.891 | 0.864 | 0.905 | 0.884 | 0.945 |
| Random Forest | 0.870 | 0.857 | 0.857 | 0.857 | 0.926 |
| Neural Network | 0.848 | 0.792 | 0.905 | 0.844 | 0.937 |

Best model: **Logistic Regression**.

## How to run
Open `DS471_Heart_Disease.ipynb` in Google Colab and click Runtime → Run all.
Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow.

## Files
- `DS471_Heart_Disease.ipynb` – full code with outputs
- `DS471_Project_Report.pdf` – final report (added after final review)
