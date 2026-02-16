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
# METHOD 1: Load using absolute path
# ============================================
def load_iris_absolute():
    """Load iris.data using absolute path"""
    data_path = r"C:\Users\gshub\OneDrive\Desktop\project 2\dataset\dataset\Iris\iris.data"
    
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    data = pd.read_csv(data_path, header=None, names=columns)
    
    # Remove any empty rows
    data = data.dropna(how="any")
    
    return data

# ============================================
# METHOD 2: Load using relative path (if script is in same folder)
# ============================================
def load_iris_relative():
    """Load iris.data using relative path from script location"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "iris.data")
    
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    data = pd.read_csv(data_path, header=None, names=columns)
    
    # Remove any empty rows
    data = data.dropna(how="any")
    
    return data

# ============================================
# METHOD 3: Load from iris.py file (alternative)
# ============================================
def load_iris_from_py():
    """Load data from iris.py file"""
    from io import StringIO
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    py_file_path = os.path.join(script_dir, "iris.py")
    
    with open(py_file_path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    # Join and treat as CSV
    s = "\n".join(lines)
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    data = pd.read_csv(StringIO(s), header=None, names=columns)
    
    return data

# ============================================
# MAIN: Load and display dataset info
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("LOADING IRIS DATASET")
    print("=" * 50)
    
    # Try loading using relative path first (most flexible)
    try:
        data = load_iris_relative()
        print("✓ Loaded using relative path method")
    except FileNotFoundError:
        try:
            data = load_iris_absolute()
            print("✓ Loaded using absolute path method")
        except FileNotFoundError:
            data = load_iris_from_py()
            print("✓ Loaded from iris.py file")
    
    print("\n" + "=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    print(f"Shape: {data.shape}")
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nLast 5 rows:")
    print(data.tail())
    print(f"\nDataset Info:")
    print(data.info())
    print(f"\nBasic Statistics:")
    print(data.describe())
    print(f"\nClass Distribution:")
    print(data['class'].value_counts())
    
    # ============================================
    # PREPARE DATA FOR TRAINING
    # ============================================
    print("\n" + "=" * 50)
    print("PREPARING DATA FOR TRAINING")
    print("=" * 50)
    
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
    # TRAIN MODEL
    # ============================================
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)
    
    model = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
    model.fit(X_train_scaled, y_train)
    print("✓ Model trained (Logistic Regression)")
    
    # ============================================
    # TEST MODEL
    # ============================================
    print("\n" + "=" * 50)
    print("TESTING MODEL")
    print("=" * 50)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # ============================================
    # VISUALIZE CONFUSION MATRIX
    # ============================================
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Iris Classification")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    print("\n" + "=" * 50)
    print("COMPLETE!")
    print("=" * 50)
