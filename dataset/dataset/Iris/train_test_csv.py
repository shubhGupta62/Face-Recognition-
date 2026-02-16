"""
TRAIN AND TEST IRIS DATASET FROM CSV FILE
==========================================
This script loads data from CSV file and trains/test a model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# LOAD CSV FILE
# ============================================
def load_iris_csv(csv_path=None):
    """Load iris dataset from CSV file"""
    if csv_path is None:
        # Get script directory and look for iris.csv
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "iris.csv")
    
    # If CSV doesn't exist, create it from iris.data
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        print("Creating CSV from iris.data...")
        
        # Try to load from iris.data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "iris.data")
        
        if os.path.exists(data_path):
            columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
            data = pd.read_csv(data_path, header=None, names=columns)
            data = data.dropna(how="any")
            
            # Save as CSV
            data.to_csv(csv_path, index=False)
            print(f"✓ Created CSV file: {csv_path}")
        else:
            raise FileNotFoundError(f"Neither iris.csv nor iris.data found!")
    
    # Load CSV file
    print(f"Loading CSV file: {csv_path}")
    data = pd.read_csv(csv_path)
    
    return data

# ============================================
# MAIN: Train and Test
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("IRIS DATASET - TRAIN AND TEST FROM CSV")
    print("=" * 60)
    
    # Load data from CSV
    data = load_iris_csv()
    
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Shape: {data.shape}")
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nDataset Info:")
    print(data.info())
    print(f"\nBasic Statistics:")
    print(data.describe())
    print(f"\nClass Distribution:")
    print(data['class'].value_counts())
    
    # ============================================
    # PREPARE DATA FOR TRAINING
    # ============================================
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 60)
    
    X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = data["class"]
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    print(f"Classes: {y.unique()}")
    
    # ============================================
    # TRAIN-TEST SPLIT
    # ============================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # ============================================
    # SCALE FEATURES
    # ============================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✓ Features scaled using StandardScaler")
    
    # ============================================
    # TRAIN MODEL (Logistic Regression)
    # ============================================
    print("\n" + "=" * 60)
    print("TRAINING MODEL - LOGISTIC REGRESSION")
    print("=" * 60)
    
    model_lr = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
    model_lr.fit(X_train_scaled, y_train)
    print("✓ Logistic Regression model trained")
    
    y_pred_lr = model_lr.predict(X_test_scaled)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    
    print(f"\nAccuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
    
    # ============================================
    # TRAIN MODEL (Random Forest)
    # ============================================
    print("\n" + "=" * 60)
    print("TRAINING MODEL - RANDOM FOREST")
    print("=" * 60)
    
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    print("✓ Random Forest model trained")
    
    y_pred_rf = model_rf.predict(X_test_scaled)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"\nAccuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
    
    # ============================================
    # COMPARE MODELS
    # ============================================
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Logistic Regression Accuracy: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
    print(f"Random Forest Accuracy:       {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
    
    # Use the best model for detailed report
    if accuracy_rf >= accuracy_lr:
        best_model = model_rf
        y_pred = y_pred_rf
        model_name = "Random Forest"
    else:
        best_model = model_lr
        y_pred = y_pred_lr
        model_name = "Logistic Regression"
    
    print(f"\nBest Model: {model_name}")
    
    # ============================================
    # DETAILED RESULTS
    # ============================================
    print("\n" + "=" * 60)
    print(f"DETAILED RESULTS - {model_name}")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # ============================================
    # VISUALIZE CONFUSION MATRIX
    # ============================================
    plt.figure(figsize=(10, 8))
    
    # Confusion Matrix
    plt.subplot(2, 1, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=best_model.classes_,
                yticklabels=best_model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    
    # Feature importance (if Random Forest)
    if model_name == "Random Forest":
        plt.subplot(2, 1, 2)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
        plt.xlabel("Importance")
        plt.title("Feature Importance - Random Forest")
    
    plt.tight_layout()
    plt.savefig("iris_results.png", dpi=150)
    print("\n✓ Results visualization saved as 'iris_results.png'")
    
    # ============================================
    # SAVE PREDICTIONS TO CSV
    # ============================================
    results_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'correct': y_test.values == y_pred
    })
    
    results_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Predictions saved to 'predictions.csv'")
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
