import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import subprocess
import cv2
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Column names (from iris.names)
columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

# Load iris.data (using absolute path)
iris_file = r"c:\Users\gshub\OneDrive\Desktop\project 2\dataset\dataset\Iris\iris.data"
data = pd.read_csv(iris_file, header=None, names=columns)

print("=" * 70)
print("IRIS DATASET ANALYSIS & MACHINE LEARNING")
print("=" * 70)

print("\n1. DATASET OVERVIEW:")
print("-" * 70)
print(f"Dataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nData types:\n{data.dtypes}")
print(f"\nClass distribution:\n{data['class'].value_counts()}")
print(f"\nDataset statistics:\n{data.describe()}")

# Prepare features and target
X = data.iloc[:, :4]  # Features
y = data.iloc[:, 4]   # Target

# Feature scaling (important for many algorithms)
print("\n2. FEATURE SCALING:")
print("-" * 70)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("Features have been standardized (mean=0, std=1)")
print(f"Scaled data sample:\n{X_scaled.head()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ============================================================
# MULTIPLE CLASSIFIERS COMPARISON
# ============================================================
print("\n3. MODEL TRAINING & EVALUATION:")
print("-" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score (K-Fold with 5 splits)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')
    
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred
    }
    
    print(f"\n{model_name}:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Cross-Validation Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"  CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
best_model = results[best_model_name]['model']
best_pred = results[best_model_name]['y_pred']

print(f"\n{'*' * 70}")
print(f"BEST MODEL: {best_model_name}")
print(f"Cross-Validation Accuracy: {results[best_model_name]['cv_mean']:.4f}")
print(f"{'*' * 70}")

# ============================================================
# DETAILED EVALUATION OF BEST MODEL
# ============================================================
print("\n4. DETAILED CLASSIFICATION REPORT (Best Model):")
print("-" * 70)
print(classification_report(y_test, best_pred))

# ============================================================
# CONFUSION MATRIX
# ============================================================
print("\n5. VISUALIZATION - CONFUSION MATRIX:")
print("-" * 70)
cm = confusion_matrix(y_test, best_pred)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')

for idx, (model_name, result) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(y_test, result['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=best_model.classes_,
                yticklabels=best_model.classes_,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name}\nAccuracy: {result['accuracy']:.4f}")

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices.png")
plt.close()
subprocess.Popen(['start', 'confusion_matrices.png'], shell=True)

# ============================================================
# MODEL COMPARISON VISUALIZATION
# ============================================================
print("\n6. VISUALIZATION - MODEL COMPARISON:")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
cv_means = [results[name]['cv_mean'] for name in model_names]

axes[0].bar(model_names, accuracies, color='skyblue', alpha=0.8, label='Test Accuracy')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim([0.9, 1.0])
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10)

# Cross-validation comparison
axes[1].bar(model_names, cv_means, color='lightcoral', alpha=0.8, label='CV Mean')
axes[1].errorbar(model_names, cv_means, 
                 yerr=[results[name]['cv_std'] for name in model_names],
                 fmt='none', color='black', capsize=5, label='CV Std Dev')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Cross-Validation Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylim([0.9, 1.0])
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(cv_means):
    axes[1].text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.close()
subprocess.Popen(['start', 'model_comparison.png'], shell=True)

# ============================================================
# FEATURE IMPORTANCE (Random Forest)
# ============================================================
print("\n7. FEATURE IMPORTANCE ANALYSIS:")
print("-" * 70)

rf_model = results['Random Forest']['model']
feature_importance = rf_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='green', alpha=0.7)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance (Random Forest Classifier)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nSaved: feature_importance.png")
plt.close()
subprocess.Popen(['start', 'feature_importance.png'], shell=True)

# ============================================================
# FEATURE DISTRIBUTION
# ============================================================
print("\n8. FEATURE DISTRIBUTION ANALYSIS:")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(X.columns):
    for iris_class in data['class'].unique():
        subset = data[data['class'] == iris_class][feature]
        axes[idx].hist(subset, alpha=0.5, label=iris_class, bins=15)
    
    axes[idx].set_xlabel(feature, fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    axes[idx].legend()

plt.suptitle('Iris Features Distribution by Class', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: feature_distribution.png")
plt.close()
subprocess.Popen(['start', 'feature_distribution.png'], shell=True)

# ============================================================
# FACE RECOGNITION SYSTEM (Using HOG Features + SVM)
# ============================================================
print("\n" + "=" * 70)
print("9. FACE RECOGNITION SYSTEM - TRAINING & PREDICTION")
print("=" * 70)

# Path to face images directory
faces_dir = r"c:\Users\gshub\OneDrive\Desktop\project 2\dataset\faces"

if os.path.exists(faces_dir):
    print(f"\nLoading faces from: {faces_dir}")
    
    # Get all person folders
    person_folders = sorted([d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))])
    print(f"Found {len(person_folders)} people: {', '.join(person_folders)}\n")
    
    # Function to extract features from an image (fast method)
    def extract_face_features(image_path):
        try:
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Resize to standard size for consistency (smaller for speed)
            img = cv2.resize(img, (64, 64))
            
            # Flatten and normalize
            features = img.flatten().astype(np.float32)
            features = features / 255.0  # Normalize to 0-1
            
            return features
        except Exception as e:
            return None
    
    # Load training data
    print("Extracting features from all face images...")
    X_faces = []
    y_faces = []
    person_image_count = {}
    total_images = sum(len([f for f in os.listdir(os.path.join(faces_dir, p)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) for p in person_folders)
    processed = 0
    
    for person_idx, person_name in enumerate(person_folders):
        person_path = os.path.join(faces_dir, person_name)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        person_image_count[person_name] = len(image_files)
        print(f"  Processing {person_name}... ({len(image_files)} images)", end='', flush=True)
        
        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            features = extract_face_features(img_path)
            processed += 1
            
            if features is not None:
                X_faces.append(features)
                y_faces.append(person_name)
        
        print(" ✓")
    
    X_faces = np.array(X_faces)
    y_faces = np.array(y_faces)
    
    print(f"\nTotal images loaded: {len(X_faces)}")
    print(f"Feature vector size: {X_faces.shape[1]}")
    
    # Split data for training and testing
    X_train_faces, X_test_faces, y_train_faces, y_test_faces = train_test_split(
        X_faces, y_faces, test_size=0.2, random_state=42, stratify=y_faces
    )
    
    print(f"Training samples: {len(X_train_faces)}")
    print(f"Testing samples: {len(X_test_faces)}")
    
    # Train SVM classifier for face recognition
    print("\nTraining Face Recognition Model (SVM)...")
    face_model = SVC(kernel='rbf', probability=True, random_state=42)
    face_model.fit(X_train_faces, y_train_faces)
    
    # Evaluate face recognition model
    y_pred_faces = face_model.predict(X_test_faces)
    face_accuracy = accuracy_score(y_test_faces, y_pred_faces)
    
    print(f"\nFace Recognition Model Accuracy: {face_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_faces, y_pred_faces))
    
    # Create confusion matrix for face recognition
    cm_faces = confusion_matrix(y_test_faces, y_pred_faces, labels=sorted(set(y_faces)))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_faces, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                xticklabels=sorted(set(y_faces)),
                yticklabels=sorted(set(y_faces)),
                cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted Person", fontsize=12, fontweight='bold')
    ax.set_ylabel("Actual Person", fontsize=12, fontweight='bold')
    ax.set_title(f"Face Recognition - Confusion Matrix\nAccuracy: {face_accuracy:.4f}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('face_recognition_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: face_recognition_confusion_matrix.png")
    plt.close()
    subprocess.Popen(['start', 'face_recognition_confusion_matrix.png'], shell=True)
    
    # ============================================================
    # SAMPLE PREDICTIONS WITH FACE IMAGES
    # ============================================================
    print("\n" + "-" * 70)
    print("10. SAMPLE FACE PREDICTIONS WITH VISUALIZATION")
    print("-" * 70)
    
    # Get sample predictions and visualize
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.ravel()
    
    sample_indices = np.random.choice(len(X_test_faces), min(9, len(X_test_faces)), replace=False)
    
    for plot_idx, test_idx in enumerate(sample_indices):
        # Get test image path
        test_sample = X_test_faces[test_idx]
        actual_person = y_test_faces[test_idx]
        
        # Predict
        predicted_person = face_model.predict([test_sample])[0]
        confidence = face_model.predict_proba([test_sample]).max()
        
        # Find and load the actual image
        person_path = os.path.join(faces_dir, actual_person)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if image_files:
            img_path = os.path.join(person_path, image_files[test_idx % len(image_files)])
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[plot_idx].imshow(img)
        
        # Add title with prediction
        is_correct = actual_person == predicted_person
        title_color = 'green' if is_correct else 'red'
        axes[plot_idx].set_title(
            f"Actual: {actual_person}\nPredicted: {predicted_person}\nConfidence: {confidence:.2f}\n{'✓ Correct' if is_correct else '✗ Wrong'}",
            fontsize=11, fontweight='bold', color=title_color
        )
        axes[plot_idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(sample_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Face Recognition - Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('face_recognition_predictions.png', dpi=300, bbox_inches='tight')
    print("\nSaved: face_recognition_predictions.png")
    print("Sample predictions displayed with confidence scores")
    plt.close()
    subprocess.Popen(['start', 'face_recognition_predictions.png'], shell=True)
    
    # ============================================================
    # SAVE TRAINED FACE RECOGNITION MODEL
    # ============================================================
    print("\n" + "-" * 70)
    print("Saving trained face recognition model...")
    model_data = {
        'model': face_model,
        'person_list': sorted(set(y_faces)),
        'feature_size': X_faces.shape[1]
    }
    with open('face_recognition_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Model saved: face_recognition_model.pkl")
    
    # Display person statistics
    print("\n" + "-" * 70)
    print("DATASET STATISTICS:")
    print("-" * 70)
    for person in sorted(set(y_faces)):
        count = np.sum(y_faces == person)
        print(f"{person:15} : {count:3} images")
    
else:
    print(f"\nFace directory not found: {faces_dir}")
    print("Skipping face recognition...")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"\n1. IRIS CLASSIFICATION:")
print(f"   Best Model: Support Vector Machine")
print(f"   Cross-Validation Accuracy: {results[best_model_name]['cv_mean']:.4f}")
print(f"   Test Set Accuracy: {results[best_model_name]['accuracy']:.4f}")

if 'face_accuracy' in locals():
    print(f"\n2. FACE RECOGNITION:")
    print(f"   Model: Support Vector Machine (SVM)")
    print(f"   Accuracy on Test Set: {face_accuracy:.4f}")
    print(f"   Total Faces: {len(X_faces)}")
    print(f"   People Recognized: {len(person_folders)}")

print(f"\nAll visualizations and models have been saved:")
print("  - confusion_matrices.png")
print("  - model_comparison.png")
print("  - feature_importance.png")
print("  - feature_distribution.png")
print("  - face_recognition_confusion_matrix.png")
print("  - face_recognition_predictions.png")
print("  - face_recognition_model.pkl")
print("\n" + "=" * 70)
