"""
HOW TO LOAD THE IRIS DATASET
============================
This file demonstrates different methods to load the iris dataset.
"""

import pandas as pd
import os

# ============================================
# METHOD 1: Absolute Path (Most Reliable)
# ============================================
print("METHOD 1: Using Absolute Path")
print("-" * 40)

data_path = r"C:\Users\gshub\OneDrive\Desktop\project 2\dataset\dataset\Iris\iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

data = pd.read_csv(data_path, header=None, names=columns)
data = data.dropna()  # Remove empty rows

print(f"✓ Loaded {len(data)} rows")
print(data.head())
print()

# ============================================
# METHOD 2: Relative Path (If script is in same folder)
# ============================================
print("METHOD 2: Using Relative Path")
print("-" * 40)

# Get current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "iris.data")

data = pd.read_csv(data_path, header=None, names=columns)
data = data.dropna()

print(f"✓ Loaded {len(data)} rows")
print(data.head())
print()

# ============================================
# METHOD 3: Change Working Directory
# ============================================
print("METHOD 3: Change Working Directory")
print("-" * 40)

# Change to the dataset directory
os.chdir(r"C:\Users\gshub\OneDrive\Desktop\project 2\dataset\dataset\Iris")
data = pd.read_csv("iris.data", header=None, names=columns)
data = data.dropna()

print(f"✓ Loaded {len(data)} rows")
print(data.head())
print()

# ============================================
# METHOD 4: Load from iris.py file
# ============================================
print("METHOD 4: Load from iris.py")
print("-" * 40)

from io import StringIO

script_dir = os.path.dirname(os.path.abspath(__file__))
py_file_path = os.path.join(script_dir, "iris.py")

with open(py_file_path, 'r', encoding='utf-8') as f:
    lines = [ln.strip() for ln in f.readlines() if ln.strip()]

s = "\n".join(lines)
data = pd.read_csv(StringIO(s), header=None, names=columns)

print(f"✓ Loaded {len(data)} rows")
print(data.head())
