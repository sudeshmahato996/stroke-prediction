"""
Stroke Prediction Machine Learning Pipeline

This script loads the stroke dataset, preprocesses it, handles class imbalance 
using SMOTE and Class Weights, trains multiple models, and evaluates their performance.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
)

# Configuration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def ensure_output_dir(dir_name="outputs"):
    """Ensure the output directory exists for saving plots and results."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_data(filepath="healthcare-dataset-stroke-data.csv"):
    """Load and perform basic EDA on the dataset."""
    print("--- Loading Data ---")
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    return df


def preprocess_data(df):
    """
    Preprocess data: handle missing values, encode categoricals, split, and scale.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, numeric_features
    """
    print("\n--- Preprocessing Data ---")
    
    # Drop irrelevant columns
    df = df.drop('id', axis=1)
    
    # Remove rare categories
    df = df[df['gender'] != 'Other'].reset_index(drop=True)
    
    # Handle missing BMI values by imputing with median
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    bmi_median = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(bmi_median)
    print(f"Filled missing BMI with median: {bmi_median:.1f}")
    
    # One-hot encoding for categorical variables
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert booleans to integers
    bool_columns = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)
    
    # Split features and target
    X = df_encoded.drop('stroke', axis=1)
    y = df_encoded['stroke']
    
    # Train-test split (stratified to maintain class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numeric features
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    feature_columns = X.columns
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, numeric_features


def apply_smote(X_train, y_train):
    """Apply SMOTE to handle the severe class imbalance on the training data."""
    print("\n--- Applying SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"SMOTE Training data shape: {X_train_smote.shape}")
    return X_train_smote, y_train_smote


def train_models(X_train, y_train, X_train_smote, y_train_smote):
    """
    Train various models using SMOTE and Class Weights to handle imbalance.
    
    Returns:
        dict: A dictionary of trained models.
    """
    print("\n--- Training Models ---")
    models = {}
    
    # Calculate scale weight for XGBoost class weight strategy
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 1. Models trained on SMOTE data
    models['LR_SMOTE'] = LogisticRegression(max_iter=1000, random_state=42)
    models['RF_SMOTE'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['XGB_SMOTE'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    
    for name in ['LR_SMOTE', 'RF_SMOTE', 'XGB_SMOTE']:
        models[name].fit(X_train_smote, y_train_smote)
        print(f"Trained {name}")

    # 2. Models trained with class weights (on original scaled data)
    models['LR_ClassWeight'] = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    models['RF_ClassWeight'] = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    models['XGB_ClassWeight'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', scale_pos_weight=scale_weight)
    
    for name in ['LR_ClassWeight', 'RF_ClassWeight', 'XGB_ClassWeight']:
        models[name].fit(X_train, y_train)
        print(f"Trained {name}")
        
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate trained models on the test set and save results."""
    print("\n--- Evaluating Models ---")
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results.append({
            'Model': name,
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)
        })
        
    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    results_df.to_csv('outputs/model_results.csv', index=False)
    
    print(results_df.to_string(index=False))
    return results_df


def plot_confusion_matrices(models, X_test, y_test):
    """Plot confusion matrices for all trained models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(name)
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png')
    plt.close()
    print("Saved confusion matrices plot to outputs/confusion_matrices.png")


def predict_interactive(model, scaler, feature_columns, numeric_features):
    """Prompt the user for patient details and predict stroke risk using the best model."""
    print("\n--- Interactive Stroke Prediction ---")
    try:
        age = float(input("Age: "))
        avg_glucose = float(input("Average glucose level (e.g. 100): "))
        bmi_input = float(input("BMI (e.g. 25.5): "))
        
        gender = input("Gender (Male/Female): ").strip().capitalize()
        hypertension = int(input("Hypertension (0=No, 1=Yes): "))
        heart_disease = int(input("Heart Disease (0=No, 1=Yes): "))
        ever_married = input("Ever Married? (Yes/No): ").strip().capitalize()
        work_type = input("Work Type (Private/Self-employed/Govt_job/children/Never_worked): ").strip()
        residence = input("Residence Type (Urban/Rural): ").strip().capitalize()
        smoking = input("Smoking Status (never smoked/formerly smoked/smokes/Unknown): ").strip()

        # Initialize input dataframe matching the training columns
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Set numeric values
        input_data['age'] = age
        input_data['avg_glucose_level'] = avg_glucose
        input_data['bmi'] = bmi_input
        input_data['hypertension'] = hypertension
        input_data['heart_disease'] = heart_disease
        
        # Set one-hot encoded categorical values
        if f'gender_{gender}' in input_data.columns:
            input_data[f'gender_{gender}'] = 1
        if f'ever_married_{ever_married}' in input_data.columns:
            input_data[f'ever_married_{ever_married}'] = 1
        if f'work_type_{work_type}' in input_data.columns:
            input_data[f'work_type_{work_type}'] = 1
        if f'Residence_type_{residence}' in input_data.columns:
            input_data[f'Residence_type_{residence}'] = 1
        if f'smoking_status_{smoking}' in input_data.columns:
            input_data[f'smoking_status_{smoking}'] = 1
            
        # Scale numeric features
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
        
        # Make prediction
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        print("\n--- Prediction Result ---")
        if pred == 1:
            print(f"HIGH RISK: Stroke predicted with probability {prob*100:.1f}%")
        else:
            print(f"LOW RISK: No stroke predicted with probability {prob*100:.1f}%")
            
    except Exception as e:
        print(f"Error parsing input: {e}")


def main():
    ensure_output_dir()
    
    # 1. Load Data
    df = load_data()
    
    # 2. Preprocess Data
    X_train, X_test, y_train, y_test, scaler, feature_cols, num_feats = preprocess_data(df)
    
    # 3. Handle Imbalance
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    
    # 4. Train Models
    models = train_models(X_train, y_train, X_train_smote, y_train_smote)
    
    # 5. Evaluate
    results_df = evaluate_models(models, X_test, y_test)
    
    # 6. Plotting
    plot_confusion_matrices(models, X_test, y_test)
    
    # 7. Interactive Prediction
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    print(f"\nBest model selected for interactive testing: {best_model_name}")
    
    # Uncomment the line below to run the interactive input prompt!
    # predict_interactive(best_model, scaler, feature_cols, num_feats)


if __name__ == "__main__":
    main()
