# Estimation of Obesity Levels Using Machine Learning


## Project Overview
This project focuses on predicting obesity levels based on health and lifestyle attributes using machine learning techniques. Multiple classification models are trained and evaluated to determine which performs best in estimating obesity categories.

The goal is to analyze how different ML algorithms perform on the same dataset and compare them using standard evaluation metrics.


## Dataset
- **Source:** Publicly available obesity dataset (e.g., UCI Machine Learning Repository / Kaggle)
- **Description:**  
  The dataset contains demographic, physical, and lifestyle-related features such as age, gender, eating habits, physical activity, and transportation methods.
- **Target Variable:**  
  Obesity level (multi-class classification)


## Machine Learning Models Used
The following models were implemented and compared:

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- XGBoost Classifier  

Each model was trained on the same training data and evaluated on a held-out test set.


## Evaluation Metrics
Model performance was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

Visual comparisons of metrics across models were also generated to analyze relative performance.


## Methodology
1. Data preprocessing and encoding  
2. Train-test split  
3. Model training  
4. Model evaluation  
5. Performance comparison across models  

All steps are implemented in a google colab for clarity and reproducibility.
