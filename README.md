# 🍷 Wine Quality Classification – SVM, KNN, ANN

This repository contains a multi-class classification project using the **Wine Quality (Red Wine)** dataset from the UCI Machine Learning Repository. The goal is to predict wine quality scores (ranging from 3 to 8) based on physicochemical test results.

---

## 📌 Project Overview

- **Dataset**: [Wine Quality – Kaggle]([https://archive.ics.uci.edu/ml/datasets/wine+quality](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset))
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
| SVM   | SMOTE Oversampled  | 56.3%    | ~0.33    |
| KNN   | Stratified Only    | 56.8%    | ~0.27    |
| KNN   | SMOTE Oversampled  | 45.4%    | ~0.25    |
| ANN   | Stratified Only    | 65.5%    | ~0.31    |
| ANN   | SMOTE Oversampled  | 61.1%    | ~0.30    |

- **Best model**: ANN, due to balanced generalization across classes
- **SMOTE** helped with minority class recall but reduced overall accuracy

---

## 🔧 RapidMiner Implementation

To validate model robustness and reproducibility, we also implemented the Artificial Neural Network (ANN) using **RapidMiner**. The same dataset (WineQT.csv) and modeling objective were applied.

### 🧭 Workflow in RapidMiner:

1. **Read CSV**  
   The dataset was loaded using the `Read CSV` operator. The `WineQT.csv` file was imported.

2. **Select Attributes**  
   The `Id` column was excluded as it provided no predictive value.

3. **Numerical to Polynomial**  
   The `quality` column (target variable) was converted to **nominal (polynomial)**, as required by classification performance evaluation in RapidMiner.

4. **Set Role**  
   The `quality` attribute was assigned the role of `label`.

5. **Normalize (Z-Transformation)**  
   All numerical features were normalized to avoid bias from feature scale in ANN training.

6. **Split Data (Stratified)**  
   The dataset was split into 80% training and 20% testing, maintaining class proportions (stratified sampling).

7. **Train Model (Neural Net)**  
   The ANN model was trained using the `Neural Net` operator with default hyperparameters.

8. **Apply Model**  
   The trained model was applied to the testing partition.

9. **Performance (Classification)**  
   Evaluation was performed using `Performance (Classification)` to compute accuracy, precision, and recall per class.

---

### 📊 Evaluation Results (ANN in RapidMiner):

- **Accuracy**: 55.02%
- **Highest Precision & Recall**: Class `5` (66.34%) and Class `6` (51.55%)
- **Minority Classes (3, 4, 8)**: Precision and recall = 0%
- The imbalance in the dataset prevented minority class recognition.
- ANN still demonstrated robust behavior on normalized and majority-class features.

---

### 📝 Observation:

- The ANN implementation in RapidMiner produced consistent results with Python (stratified version, without SMOTE).
- Data preprocessing steps (normalization, label conversion, stratified split) are essential to replicate model performance across platforms.

---

## 🧰 Tools & Libraries

- **Python**: `pandas`, `numpy`, `scikit-learn`, `keras`, `seaborn`, `matplotlib`
- **RapidMiner**: Parallel implementation of ANN for evaluation consistency

---

## 📁 File Structure

