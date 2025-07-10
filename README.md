# 🍷 Wine Quality Classification – SVM, KNN, ANN

This repository contains a multi-class classification project using the **Wine Quality (Red Wine)** dataset from the UCI Machine Learning Repository. The goal is to predict wine quality scores (ranging from 3 to 8) based on physicochemical test results.

---

## 📌 Project Overview

- **Dataset**: [Wine Quality – UCI Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Data Size**: 1143 samples, 11 features + 1 target (`quality`)
- **Target**: Multi-class classification (`quality` ∈ [3, 4, 5, 6, 7, 8])
- **Problem**: Imbalanced class distribution with majority in class 5 and 6

---

## 🧪 Methods and Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked for missing values (none)
- Visualized feature distributions and correlations
- Identified class imbalance in target variable

### 2. Data Preprocessing
- Removed `Id` column
- Normalized features using `StandardScaler`
- Performed stratified train-test split (80:20)

### 3. Modeling
Implemented and compared three classifiers:
- **SVM** – Support Vector Machine (Scikit-learn)
- **KNN** – K-Nearest Neighbors (Scikit-learn)
- **ANN** – Artificial Neural Network (Keras)

Each model was evaluated under:
- **Stratified sampling only**
- **SMOTE oversampling** (on training set)

### 4. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (macro average)
- Confusion Matrix
- Optional: 5-Fold Cross Validation (SVM, KNN, ANN)

---

## 🏁 Results Summary

| Model | Strategy           | Accuracy | Macro F1 |
|-------|--------------------|----------|----------|
| SVM   | Stratified Only    | 66.4%    | ~0.31    |
| KNN   | Stratified Only    | 56.8%    | ~0.27    |
| ANN   | Stratified Only    | 62.4%    | ~0.29    |
| SVM   | SMOTE Oversampled  | 56.3%    | ~0.33    |
| KNN   | SMOTE Oversampled  | 45.4%    | ~0.25    |
| ANN   | SMOTE Oversampled  | 58.9%    | ~0.30    |

- **Best model**: ANN, due to balanced generalization across classes
- **SMOTE** helped with minority class recall but reduced overall accuracy

---

## 🧰 Tools & Libraries

- **Python**: `pandas`, `numpy`, `scikit-learn`, `keras`, `seaborn`, `matplotlib`
- **RapidMiner**: Parallel implementation of ANN for evaluation consistency

---

## 📁 File Structure

