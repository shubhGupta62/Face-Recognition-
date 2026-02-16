"""
CONVERT IRIS DATA TO CSV FILE
==============================
This script converts iris.data to a proper CSV file format.
"""

import pandas as pd
import os

# ============================================
# Load iris.data and convert to CSV
# ============================================

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "iris.data")
csv_path = os.path.join(script_dir, "iris.csv")

# Column names
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Load the data
print("Loading iris.data...")
data = pd.read_csv(data_path, header=None, names=columns)

# Remove empty rows
data = data.dropna(how="any")

print(f"✓ Loaded {len(data)} rows")
print(f"\nFirst few rows:")
print(data.head())

# Save as CSV
print(f"\nSaving to CSV file: {csv_path}")
data.to_csv(csv_path, index=False)

print(f"✓ Successfully saved as 'iris.csv'")
print(f"\nCSV file location: {csv_path}")
print(f"\nCSV file info:")
print(f"  - Rows: {len(data)}")
print(f"  - Columns: {len(data.columns)}")
print(f"  - File size: {os.path.getsize(csv_path)} bytes")
