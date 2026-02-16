# ============================================
# PCA + ANN REGRESSION ON YOUTUBE ADVIEW DATA
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 1. LOAD DATASET
# ============================================

column_names = [
    'vidid',
    'adview',
    'views',
    'likes',
    'dislikes',
    'comment',
    'published',
    'duration',
    'category'
]

df = pd.read_csv(
    "C:\Users\gshub\Downloads\test_lyst1717074532669 (3) (1).csv",   # change name if needed
    header=None,
    names=column_names
)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head())

# ============================================
# 2. FEATURE SELECTION
# ============================================

features = ['views', 'likes', 'dislikes', 'comment', 'duration']
target = 'adview'

X = df[features]
y = df[target]

# ============================================
# 3. CONVERT DURATION (PT#M#S → seconds)
# ============================================

def convert_duration(d):
    minutes, seconds = 0, 0
    d = str(d)
    if 'M' in d:
        minutes = int(d.split('M')[0].replace('PT', ''))
    if 'S' in d:
        seconds = int(d.split('M')[-1].replace('S', ''))
    return minutes * 60 + seconds

X['duration'] = X['duration'].apply(convert_duration)

# ============================================
# 4. HANDLE MISSING VALUES
# ============================================

X = X.fillna(X.mean())

# ============================================
# 5. TRAIN-TEST SPLIT (60-40)
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# ============================================
# 6. STANDARDIZATION
# ============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 7. PCA + ANN (REGRESSION)
# ============================================

k_values = [2, 3, 4, 5]
rmse_list = []
r2_list = []

for k in k_values:
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    ann = MLPRegressor(
        hidden_layer_sizes=(50,),
        max_iter=1000,
        random_state=42
    )

    ann.fit(X_train_pca, y_train)
    y_pred = ann.predict(X_test_pca)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f"k = {k} | RMSE = {rmse:.2f} | R² = {r2:.4f}")

# ============================================
# 8. PLOT RESULTS
# ============================================

plt.figure(figsize=(8,5))
plt.plot(k_values, rmse_list, marker='o')
plt.xlabel("Number of PCA Components (k)")
plt.ylabel("RMSE")
plt.title("RMSE vs PCA Components")
plt.grid()
plt.show()

# ============================================
# 9. BEST MODEL
# ============================================

best_k = k_values[np.argmin(rmse_list)]
print("Best number of PCA components:", best_k)
