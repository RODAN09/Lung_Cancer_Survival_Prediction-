# model_train_full.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# =============================
# 1. Load Full Dataset
# =============================
df = pd.read_csv("dataset_med.csv")
print("✅ Loaded full dataset:", df.shape)

# Convert dates
df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
df["end_treatment_date"] = pd.to_datetime(df["end_treatment_date"], errors='coerce')

# Feature engineering
df["treatment_duration_days"] = (df["end_treatment_date"] - df["diagnosis_date"]).dt.days
df["diagnosis_year"] = df["diagnosis_date"].dt.year
df["diagnosis_month"] = df["diagnosis_date"].dt.month

# Drop unused or ID columns
df.drop(["id", "diagnosis_date", "end_treatment_date"], axis=1, inplace=True)

# Encode categorical features
cat_cols = ["gender", "country", "cancer_stage", "family_history", "smoking_status", "treatment_type"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Split into features/target
X = df.drop("survived", axis=1)
y = df["survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 2. Train LightGBM
# =============================
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

print("🚀 Training LightGBM on full dataset...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)


# =============================
# 3. Evaluate
# =============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n✅ Model Evaluation on Full Dataset:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Top 10 feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(importances.head(10))

# =============================
# 4. Save Model
# =============================
joblib.dump(model, "lung_cancer_model_full.pkl")
print("\n🎯 Model saved as 'lung_cancer_model_full.pkl'")
