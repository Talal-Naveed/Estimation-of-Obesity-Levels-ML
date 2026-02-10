# =======================================================
# 1. Dataset Loading
# =======================================================

from ucimlrepo import fetch_ucirepo

# Fetch dataset from the UCI Machine Learning Repository
dataset = fetch_ucirepo(id=544)

# Separate features and target variables
X = dataset.data.features
y = dataset.data.targets

# Dataset metadata and variable descriptions available on UCI repository

# ===================================================
# 2. Data Cleaning and Preprocessing
# ===================================================

import pandas as pd

# Combine features and target into a single DataFrame
df = pd.concat([X, y], axis=1)

#  Remove missing values
df.dropna(inplace=True)

# Remove duplicate rows if any exist
if df.duplicated().any()
    df.drop_duplicates(inplace=True)


 # ===================================================
# 3. Feature Encoding
# ===================================================  

from sklearn.preprocessing import LabelEncoder

# Encode binary categorical features
df['Gender'], c =pd.factorize(df['Gender'])
df['family_history_with_overweight'], c = pd.factorize(df['family_history_with_overweight'])
# Correcting column names based on the dataframe
df['FAVC'], c = pd.factorize(df['FAVC']) # Correcting FCOHCF to FAVC
df['SMOKE'], c = pd.factorize(df['SMOKE']) # Correcting Smoke to SMOKE
df['SCC'], c = pd.factorize(df['SCC']) # Correcting Calorie_Consump_Monitoring to SCC

# Initialize label encoder
le = LabelEncoder()

# Label encode other categorical columns
df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])
df['CAEC'] = le.fit_transform(df['CAEC'])
df['CALC'] = le.fit_transform(df['CALC'])
df['MTRANS'] = le.fit_transform(df['MTRANS'])


# =====================================================
# 4. Feature-Target Separation
# =====================================================

# Separate independent features (X) and target variable (y)
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']


# ========================================================
# 5. Train-Test Split
# ========================================================

from sklearn.model_selection import train_test_split

# Split dataset into training and test set
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)


# =======================================================
# 6.1 Logistic Regression Model
# =======================================================


import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Suppress only the specific convergence warning
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize and train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(train_X, train_y)

# Generate predictions
pred_logistic = logistic_model.predict(test_X)


# =================================================
# 6.1.1 Logistic Regression - Confusion Matrix
# =================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define class names corresponding to the encoded labels (0-6)
# Based on the value_counts() output for NObeyesdad and the variable information
# 0: Insufficient Weight
# 1: Normal Weight
# 2: Overweight Level I
# 3: Overweight Level II
# 4: Obesity Type I
# 5: Obesity Type II
# 6: Obesity Type III
class_names = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']


# Generate the confusion matrix 
cm_logistic = confusion_matrix(test_y, pred_logistic)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logistic, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names) # Use class_names for labels
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Logistic Regression Model')
plt.tight_layout()
plt.show()


# =======================================================
# 6.2 Decision Tree Model
# =======================================================

# Creating DT model
from sklearn.tree import DecisionTreeClassifier

# Initialize and train Decision Tree classifier
dt_model = DecisionTreeClassifier(max_depth=4, random_state=40)
dt_model.fit(train_X, train_y)

# Predicting the model
pred_dt = dt_model.predict(test_X)


# =============================================
# 6.2.1 Decision Tree - Confusion Matrix
# =============================================


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Class names based on label encoding (reusing the list defined earlier)
# class_names = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
#                'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']


# Generate the confusion matrix for the Decision Tree model
cm_dt = confusion_matrix(test_y, pred_dt)

# Plot the confusion matrix for Decision Tree as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Decision Tree Model')
plt.tight_layout()
plt.show()


# =================================================================
# 6.3 Random Forest Model
# =================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train Random Forest classifier
rf_model = RandomForestClassifier(random_state=40)
rf_model.fit(train_X, train_y)

# Generate predictions
pred_rf = rf_model.predict(test_X)


# ==================================================
# 6.3.1 Random Forest - Confusion Matrix
# ==================================================


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Class names based on label encoding
class_names = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
               'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

# Generate the confusion matrix 
cm_rf = confusion_matrix(test_y, pred_rf)

# Plot the confusion matrix for Random Forest as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Random Forest Model')
plt.tight_layout()
plt.show()


# =======================================================
# 6.4 XGBoost Model
# =======================================================

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train XGBoost classifier
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=40, use_label_encoder=False)
xgb_model.fit(train_X, train_y)

# Generate predictions
pred_xgb = xgb_model.predict(test_X)


# ========================================================
# 6.4.1 XGBoost - Confusion Matrix
# ========================================================

# Generate the confusion matrix for the XGBoost model
cm_xgb = confusion_matrix(test_y, pred_xgb)

# Plot the confusion matrix for XGBoost as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for XGBoost Model')
plt.tight_layout()
plt.show()


# ======================================================
# 7. Model Performance Comparison
# ======================================================

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Calculate precision, recall, and f1-score for each model
precision_logistic = precision_score(test_y, pred_logistic, average='weighted')
recall_logistic = recall_score(test_y, pred_logistic, average='weighted')
f1_logistic = f1_score(test_y, pred_logistic, average='weighted')

precision_dt = precision_score(test_y, pred_dt, average='weighted')
recall_dt = recall_score(test_y, pred_dt, average='weighted')
f1_dt = f1_score(test_y, pred_dt, average='weighted')

precision_rf = precision_score(test_y, pred_rf, average='weighted')
recall_rf = recall_score(test_y, pred_rf, average='weighted')
f1_rf = f1_score(test_y, pred_rf, average='weighted')

precision_xgb = precision_score(test_y, pred_xgb, average='weighted')
recall_xgb = recall_score(test_y, pred_xgb, average='weighted')
f1_xgb = f1_score(test_y, pred_xgb, average='weighted')


# Creating dataframe of performance metrics
result_metrics = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(test_y, pred_logistic),
        accuracy_score(test_y, pred_dt),
        accuracy_score(test_y, pred_rf),
        accuracy_score(test_y, pred_xgb)
    ],
    'Precision': [
        precision_logistic,
        precision_dt,
        precision_rf,
        precision_xgb
    ],
    'Recall': [
        recall_logistic,
        recall_dt,
        recall_rf,
        recall_xgb
    ],
    'F1-Score': [
        f1_logistic,
        f1_dt,
        f1_rf,
        f1_xgb
    ]
})

# Format metrics to two decimal places
result_metrics['Accuracy'] = result_metrics['Accuracy'].map('{:.2f}'.format)
result_metrics['Precision'] = result_metrics['Precision'].map('{:.2f}'.format)
result_metrics['Recall'] = result_metrics['Recall'].map('{:.2f}'.format)
result_metrics['F1-Score'] = result_metrics['F1-Score'].map('{:.2f}'.format)

# Display comparison table
display(result_metrics)


# ===================================================
# 7.1 Visual Comparison of Model Performance
# ===================================================

import matplotlib.pyplot as plt

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

# Convert the metrics columns to numeric types for plotting
result_metrics[metrics] = result_metrics[metrics].astype(float)

for metric in metrics:
    plt.figure(figsize=(8, 5))
    # Use the correct DataFrame name and plot the metric for each model
    plt.bar(result_metrics['Model'], result_metrics[metric], color='skyblue', edgecolor='black')
    plt.title(f"{metric} Comparison Across Models")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
