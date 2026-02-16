"""
SIMPLE CSV LOADING EXAMPLES
===========================
Quick examples for loading CSV files
"""

import pandas as pd
import os

# ============================================
# METHOD 1: Load CSV with absolute path
# ============================================
print("METHOD 1: Load CSV with absolute path")
print("-" * 50)

csv_path = r"C:\Users\gshub\OneDrive\Desktop\project 2\dataset\dataset\Iris\iris.csv"

# Check if CSV exists, if not create it
if not os.path.exists(csv_path):
    print("CSV file doesn't exist. Creating from iris.data...")
    # Load from iris.data first
    data_path = r"C:\Users\gshub\OneDrive\Desktop\project 2\dataset\dataset\Iris\iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    data = pd.read_csv(data_path, header=None, names=columns)
    data = data.dropna()
    # Save as CSV
    data.to_csv(csv_path, index=False)
    print(f"✓ Created {csv_path}")

# Load CSV file
data = pd.read_csv(csv_path)
print(f"✓ Loaded CSV: {len(data)} rows")
print(data.head())
print()

# ============================================
# METHOD 2: Load CSV with relative path
# ============================================
print("METHOD 2: Load CSV with relative path")
print("-" * 50)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "iris.csv")

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    print(f"✓ Loaded CSV: {len(data)} rows")
    print(data.head())
else:
    print(f"File not found: {csv_path}")
print()

# ============================================
# METHOD 3: Load CSV and specify columns
# ============================================
print("METHOD 3: Load CSV with column specification")
print("-" * 50)

csv_path = r"C:\Users\gshub\OneDrive\Desktop\project 2\dataset\dataset\Iris\iris.csv"

# If CSV already has headers, pandas will use them automatically
# But you can also specify them explicitly
data = pd.read_csv(csv_path)
print(f"✓ Loaded CSV with columns: {list(data.columns)}")
print(data.head())
print()

# ============================================
# METHOD 4: Save DataFrame to CSV
# ============================================
print("METHOD 4: Save DataFrame to CSV")
print("-" * 50)

# Example: Save a subset to new CSV
subset = data.head(10)
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris_sample.csv")
subset.to_csv(output_path, index=False)
print(f"✓ Saved sample to: {output_path}")
print(f"  Rows saved: {len(subset)}")
