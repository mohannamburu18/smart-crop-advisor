"""
Crop Recommendation Model Training
Dataset: Crop_recommendation.csv
Features: N, P, K, temperature, humidity, ph, rainfall
Target: crop label (22 crops)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = "data/Crop_recommendation.csv"   # update path if needed
MODEL_DIR  = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Load & Explore ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nSample:\n", df.head())
print("\nClass Distribution:\n", df['label'].value_counts())
print("\nNull Values:\n", df.isnull().sum())
print("\nBasic Stats:\n", df.describe())

# ─── Preprocessing ────────────────────────────────────────────────────────────
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─── Model Comparison ────────────────────────────────────────────────────────
models = {
    'RandomForest':      RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting':  GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM':               SVC(kernel='rbf', probability=True, random_state=42),
    'KNN':               KNeighborsClassifier(n_neighbors=5)
}

results = {}
best_model = None
best_acc   = 0
best_name  = ''

print("\n=== Model Comparison ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(model, X_scaled, y_enc, cv=5, scoring='accuracy').mean()
    results[name] = {'test_accuracy': acc, 'cv_accuracy': cv}
    print(f"{name:20s} | Test: {acc*100:.2f}% | CV: {cv*100:.2f}%")
    if acc > best_acc:
        best_acc   = acc
        best_model = model
        best_name  = name

print(f"\n✅ Best Model: {best_name} ({best_acc*100:.2f}%)")

# ─── Detailed Report for Best Model ──────────────────────────────────────────
y_pred_best = best_model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# ─── Confusion Matrix Plot ────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Greens')
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{MODEL_DIR}/crop_confusion_matrix.png')
print(f"Confusion matrix saved.")

# ─── Feature Importance (if RandomForest) ────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    fi.plot(kind='bar', color='green')
    plt.title('Feature Importance - Crop Recommendation')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/crop_feature_importance.png')
    print("Feature importance saved.")
    print("\nFeature Importances:\n", fi)

# ─── Save Artifacts ──────────────────────────────────────────────────────────
joblib.dump(best_model, f'{MODEL_DIR}/crop_model.pkl')
joblib.dump(scaler,     f'{MODEL_DIR}/crop_scaler.pkl')
joblib.dump(le,         f'{MODEL_DIR}/crop_label_encoder.pkl')

print(f"\n✅ Saved: crop_model.pkl | crop_scaler.pkl | crop_label_encoder.pkl")
print(f"Classes: {list(le.classes_)}")
