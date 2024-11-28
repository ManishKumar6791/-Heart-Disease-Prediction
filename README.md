# Heart Disease Prediction

This repository contains a machine learning model to predict the likelihood of heart disease using the Cleveland Heart Disease dataset. The goal is to apply various machine learning algorithms and compare their performance using metrics such as accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC curve, and more.

## Dataset

The **Cleveland Heart Disease Dataset** contains 14 attributes, including 13 features and 1 target variable. The features include both numerical and categorical values. Below is a detailed description of the features:

1. **Age**: Patient's age in years (Numeric)
2. **Sex**: Gender of the patient (Male: 1; Female: 0) (Nominal)
3. **cp**: Type of chest pain experienced by the patient, categorized into 4 types:
   - 0: Typical Angina
   - 1: Atypical Angina
   - 2: Non-anginal Pain
   - 3: Asymptomatic (Nominal)
4. **trestbps**: Patient's level of blood pressure at rest in mm/Hg (Numeric)
5. **chol**: Serum cholesterol in mg/dl (Numeric)
6. **fbs**: Fasting blood sugar levels > 120 mg/dl (1 for True, 0 for False) (Nominal)
7. **restecg**: Result of electrocardiogram while at rest, represented by 3 distinct values:
   - 0: Normal
   - 1: ST-T Wave Abnormality (T-wave inversions and/or ST elevation or depression of > 0.05 mV)
   - 2: Probable or definite left ventricular hypertrophy by Estes' criteria (Nominal)
8. **thalach**: Maximum heart rate achieved (Numeric)
9. **exang**: Exercise-induced angina (1 for Yes, 0 for No) (Nominal)
10. **oldpeak**: Exercise-induced ST depression in relation to the resting state (Numeric)
11. **slope**: ST segment slope during peak exercise:
    - 0: Up sloping
    - 1: Flat
    - 2: Down sloping (Nominal)
12. **ca**: The number of major vessels (0-3) (Nominal)
13. **thal**: A blood disorder called thalassemia:
    - 0: NULL
    - 1: Normal blood flow
    - 2: Fixed defect (no blood flow in part of the heart)
    - 3: Reversible defect (abnormal blood flow observed) (Nominal)
14. **target**: The target variable, indicating whether the patient is suffering from heart disease:
    - 1: Patient has heart disease
    - 0: Patient is normal (Target Variable)

You can access the dataset here: [Heart Disease Cleveland Dataset on Kaggle](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)

## Project Overview

- **Dataset**: Cleveland Heart Disease Dataset
  - **Use Case**: Predict the presence of heart disease based on patient data.
  - **Key Features**: Age, cholesterol levels, maximum heart rate, chest pain type, etc.
  
- **Techniques Used**:
  - Data Preprocessing: Handle missing values, categorical variable encoding, scaling features.
  - Feature Selection: Using Principal Component Analysis (PCA) to reduce dimensionality.
  - Model Training: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and XGBoost.
  - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, and AUC.

## Key Features

- **Data Preprocessing**: 
  - Missing value imputation
  - Encoding categorical features
  - Feature scaling
- **Feature Engineering**: 
  - PCA-based dimensionality reduction for model optimization.
- **Model Comparison**: 
  - Compare models using both PCA and non-PCA data.
  - Evaluate models using a wide range of metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, AUC).
  
## Installation

To run this project locally, follow these steps:

### Clone the repository

```bash
git clone https://github.com/ManishKumar6791/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
