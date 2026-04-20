import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix)

os.makedirs("model",   exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/real_traffic_data.csv")
df = df.dropna()
print(f"  Loaded {len(df):,} rows")

# ── Target ────────────────────────────────────────────────────
TARGET = "is_congested_next"
y = df[TARGET]

# ── Encode edge ───────────────────────────────────────────────
encoder = LabelEncoder()
df["edge_encoded"] = encoder.fit_transform(df["edge_id"])

# ── Feature engineering ───────────────────────────────────────
print("Engineering features...")

# Time-cyclic features (helps model understand that 23:55 is close to 00:00)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Rolling features per road (sorted by time within each day)
df = df.sort_values(["edge_id", "day_of_week", "hour", "minute"])
df["speed_rolling_3"]  = df.groupby("edge_id")["avg_speed_kmh"].transform(
    lambda x: x.rolling(3, min_periods=1).mean())
df["wait_rolling_3"]   = df.groupby("edge_id")["waiting_time"].transform(
    lambda x: x.rolling(3, min_periods=1).mean())
df["cr_rolling_3"]     = df.groupby("edge_id")["congestion_ratio"].transform(
    lambda x: x.rolling(3, min_periods=1).mean())

# Speed trend (current vs rolling avg — negative means decelerating)
df["speed_trend"] = df["avg_speed_kmh"] - df["speed_rolling_3"]

features = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "edge_encoded", "length_m",
    "vehicle_count", "avg_speed_kmh", "waiting_time",
    "occupancy", "congestion_ratio",
    "speed_ratio", "density",
    "speed_rolling_3", "wait_rolling_3", "cr_rolling_3",
    "speed_trend",
]

X = df[features]
y = df[TARGET]

print(f"Total rows           : {len(df):,}")
print(f"Congested-next rows  : {y.sum():,}")
print(f"Normal-next rows     : {(y == 0).sum():,}")
print(f"Features             : {len(features)}")

# ── Train / test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Train ─────────────────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)

print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Normal", "Congested-Next"]))

# ── Graph — exact style from your screenshot ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Random Forest Model — Congestion Prediction Results",
             fontsize=14, fontweight="bold")

# Left — Feature importance horizontal bar chart
importances = pd.Series(model.feature_importances_,
                        index=features).sort_values(ascending=True)
importances.plot(kind="barh", ax=axes[0], color="#3B8BD4", edgecolor="none")
axes[0].set_title("Feature Importance", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Importance Score")
axes[0].set_xlim(0, importances.max() * 1.15)
axes[0].tick_params(axis="y", labelsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Right — Confusion matrix heatmap (matches your screenshot style)
im = axes[1].imshow(cm, interpolation="nearest", cmap="Blues")
axes[1].set_title("Confusion Matrix", fontsize=12, fontweight="bold")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Normal", "Congested-Next"], fontsize=11)
axes[1].set_yticklabels(["Normal", "Congested-Next"], fontsize=11)
axes[1].set_xlabel("Predicted", fontsize=11)
axes[1].set_ylabel("Actual",    fontsize=11)

# Color bar
plt.colorbar(im, ax=axes[1])

# Cell labels — white text on dark cells, black on light
for i in range(2):
    for j in range(2):
        val       = cm[i, j]
        threshold = cm.max() / 2.0
        color     = "white" if val > threshold else "black"
        axes[1].text(j, i, str(val),
                     ha="center", va="center",
                     color=color, fontsize=18, fontweight="bold")

plt.tight_layout()
plt.savefig("results/model_evaluation.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/model_evaluation.png")
plt.close()

# ── Save model files ──────────────────────────────────────────
joblib.dump(model,    "model/traffic_model.pkl")
joblib.dump(encoder,  "model/label_encoder.pkl")
joblib.dump(features, "model/feature_names.pkl")
print("Saved: model/traffic_model.pkl")
print("Saved: model/label_encoder.pkl")
print("Saved: model/feature_names.pkl")
print("\nModel, encoder, and feature names saved successfully!")
print("You are now fully synced and ready to run optimizer.py")
