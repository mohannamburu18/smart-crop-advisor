"""
Fertilizer Recommendation Model Training
Dataset: Fertilizer Prediction.csv
Features: Temperature, Humidity, Moisture, Soil Type, Crop Type, N, K, P
Target: Fertilizer Name (7 types)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/Fertilizer Prediction.csv"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Load & Clean ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()   # remove whitespace from column names

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSample:\n", df.head())
print("\nSoil Types:", df['Soil Type'].unique())
print("Crop Types:", df['Crop Type'].unique())
print("Fertilizers:", df['Fertilizer Name'].unique())

# ─── Encode Categoricals ─────────────────────────────────────────────────────
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df['Soil Type']       = le_soil.fit_transform(df['Soil Type'])
df['Crop Type']       = le_crop.fit_transform(df['Crop Type'])
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

# ─── Features & Target ───────────────────────────────────────────────────────
X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ─── Train ───────────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
cv  = cross_val_score(model, X_scaled, y, cv=5).mean()

print(f"\nTest Accuracy : {acc*100:.2f}%")
print(f"CV Accuracy   : {cv*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, model.predict(X_test), target_names=le_fert.classes_))

# ─── Save ────────────────────────────────────────────────────────────────────
joblib.dump(model,   f'{MODEL_DIR}/fertilizer_model.pkl')
joblib.dump(scaler,  f'{MODEL_DIR}/fertilizer_scaler.pkl')
joblib.dump(le_soil, f'{MODEL_DIR}/fertilizer_le_soil.pkl')
joblib.dump(le_crop, f'{MODEL_DIR}/fertilizer_le_crop.pkl')
joblib.dump(le_fert, f'{MODEL_DIR}/fertilizer_le_fert.pkl')

print("\n✅ All fertilizer model artifacts saved!")
print(f"Soil Types  : {list(le_soil.classes_)}")
print(f"Crop Types  : {list(le_crop.classes_)}")
print(f"Fertilizers : {list(le_fert.classes_)}")
