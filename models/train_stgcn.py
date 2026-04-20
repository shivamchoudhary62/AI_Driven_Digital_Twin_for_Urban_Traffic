"""
train_stgcn.py
──────────────
Trains the STGCN model on the expanded real traffic dataset (80K+ rows)
to predict future traffic states across Vadodara's road network.

Usage:
    python models/train_stgcn.py

Input:  data/real_traffic_data.csv  (80,640 rows, 10 roads)
Output: model/stgcn_model.pt       (trained STGCN weights)
        results/stgcn_evaluation.png
"""

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.stgcn_model import STGCN

# ── Config ────────────────────────────────────────────────────
FEATURES      = ["avg_speed_kmh", "vehicle_count", "occupancy", "congestion_ratio"]
NUM_FEATURES  = len(FEATURES)
OUT_FEATURES  = NUM_FEATURES
TIME_WINDOW   = 12            # look-back: 12 steps (= 6 hours at 30-min intervals)
BATCH_SIZE    = 32
EPOCHS        = 80
LR            = 0.001
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── 1. Load data ─────────────────────────────────────────────
print("Loading real traffic dataset...")
df = pd.read_csv("data/real_traffic_data.csv")
df = df.dropna()
print(f"  Loaded {len(df):,} rows, {df['edge_id'].nunique()} roads")

edges = sorted(df["edge_id"].unique())
edge_to_idx = {e: i for i, e in enumerate(edges)}
num_nodes = len(edges)

# Road name mapping for display
road_names = {
    "8627861908_318141150":  "Race Course Road",
    "8451855738_8451855739": "RC Dutt Road",
    "8556786766_8659203468": "Jetalpur Road",
    "5302825179_8485898828": "Old Padra Road",
    "317076202_8526975514":  "Gotri Road",
    "7865698072_7865698075": "Manjalpur Gate Rd",
    "8560186021_327366061":  "New Sama Road",
    "320707451_320707452":   "Natubhai Circle",
    "8527120663_2345133351": "Dandia Bazaar Rd",
    "8527411421_2346917835": "Raopura Road",
}

print(f"  Roads: {[road_names.get(e, e) for e in edges]}")


# ── 2. Build adjacency matrix ────────────────────────────────
# Vadodara road connectivity (which roads connect/intersect)
# Based on real geography: roads that share intersections or are nearby
connectivity = {
    "Race Course Road":  ["RC Dutt Road", "Old Padra Road", "Natubhai Circle"],
    "RC Dutt Road":      ["Race Course Road", "Jetalpur Road", "Dandia Bazaar Rd"],
    "Jetalpur Road":     ["RC Dutt Road", "New Sama Road", "Gotri Road"],
    "Old Padra Road":    ["Race Course Road", "Manjalpur Gate Rd"],
    "Gotri Road":        ["Jetalpur Road", "New Sama Road"],
    "Manjalpur Gate Rd": ["Old Padra Road", "Raopura Road"],
    "New Sama Road":     ["Jetalpur Road", "Gotri Road"],
    "Natubhai Circle":   ["Race Course Road", "Dandia Bazaar Rd", "Raopura Road"],
    "Dandia Bazaar Rd":  ["RC Dutt Road", "Natubhai Circle", "Raopura Road"],
    "Raopura Road":      ["Natubhai Circle", "Dandia Bazaar Rd", "Manjalpur Gate Rd"],
}

# Reverse lookup: name → edge_id
name_to_edge = {v: k for k, v in road_names.items()}

adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
for road, neighbors in connectivity.items():
    eid = name_to_edge.get(road)
    if eid and eid in edge_to_idx:
        i = edge_to_idx[eid]
        for nbr in neighbors:
            neid = name_to_edge.get(nbr)
            if neid and neid in edge_to_idx:
                j = edge_to_idx[neid]
                adj[i][j] = 1.0
                adj[j][i] = 1.0

# Self-loops
for i in range(num_nodes):
    adj[i][i] = 1.0

# Symmetric normalization: D^{-1/2} A D^{-1/2}
degree = adj.sum(axis=1)
d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

adj_tensor = torch.tensor(adj_norm, dtype=torch.float32).to(DEVICE)
print(f"  Adjacency: {num_nodes} nodes, {int((adj > 0).sum())} connections")


# ── 3. Build time-series tensor ──────────────────────────────
print("\nBuilding traffic state tensor...")

# Create a unique time index: day * 288 + hour * 12 + minute / 5
# But our data has 30-min intervals (288 slots per week per road)
# Sort by time, then by edge
df = df.sort_values(["day_of_week", "hour", "minute", "edge_id"]).reset_index(drop=True)

# Group by time slot
df["time_slot"] = df["day_of_week"] * 1440 + df["hour"] * 60 + df["minute"]
time_slots = sorted(df["time_slot"].unique())
print(f"  Unique time slots: {len(time_slots)}")

# Build (T, N, F) tensor
data_list = []
for ts in time_slots:
    slot_df = df[df["time_slot"] == ts]
    feat_matrix = np.zeros((num_nodes, NUM_FEATURES), dtype=np.float32)
    for _, row in slot_df.iterrows():
        eid = row["edge_id"]
        if eid in edge_to_idx:
            idx = edge_to_idx[eid]
            feat_matrix[idx] = [row[f] for f in FEATURES]
    data_list.append(feat_matrix)

traffic_data = np.array(data_list)
print(f"  Traffic tensor shape: {traffic_data.shape}  (T={len(time_slots)}, N={num_nodes}, F={NUM_FEATURES})")

# Normalize (z-score per feature)
means = traffic_data.reshape(-1, NUM_FEATURES).mean(axis=0)
stds  = traffic_data.reshape(-1, NUM_FEATURES).std(axis=0) + 1e-8
traffic_norm = (traffic_data - means) / stds

np.savez("model/stgcn_norm_params.npz", means=means, stds=stds)
print(f"  Normalization params saved")


# ── 4. Sliding-window dataset ────────────────────────────────
class TrafficDataset(Dataset):
    def __init__(self, data, window=12):
        self.data = data
        self.window = window
        self.n_samples = len(data) - window

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window]
        y = self.data[idx + self.window]
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)


# Temporal split: 80% train, 20% test
split_idx = int(len(traffic_norm) * 0.8)
train_data = traffic_norm[:split_idx]
test_data  = traffic_norm[split_idx - TIME_WINDOW:]

train_dataset = TrafficDataset(train_data, TIME_WINDOW)
test_dataset  = TrafficDataset(test_data, TIME_WINDOW)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n  Train samples: {len(train_dataset):,}")
print(f"  Test samples:  {len(test_dataset):,}")


# ── 5. Train STGCN ───────────────────────────────────────────
print(f"\nTraining STGCN on {DEVICE} ({EPOCHS} epochs)...")
model = STGCN(
    num_nodes=num_nodes,
    in_features=NUM_FEATURES,
    out_features=OUT_FEATURES,
    time_steps=TIME_WINDOW,
    hidden_channels=64,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

train_losses = []
test_losses  = []
best_test_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        pred = model(x_batch, adj_tensor)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_train = epoch_loss / len(train_loader)
    train_losses.append(avg_train)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred = model(x_batch, adj_tensor)
            test_loss += criterion(pred, y_batch).item()

    avg_test = test_loss / len(test_loader)
    test_losses.append(avg_test)

    # Save best model
    if avg_test < best_test_loss:
        best_test_loss = avg_test
        torch.save(model.state_dict(), "model/stgcn_best.pt")

    scheduler.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"Train MSE: {avg_train:.6f}  "
              f"Test MSE: {avg_test:.6f}")

# Load best weights
model.load_state_dict(torch.load("model/stgcn_best.pt", weights_only=True))


# ── 6. Evaluate ──────────────────────────────────────────────
print("\nEvaluating on test set...")
model.eval()
all_preds = []
all_actuals = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        pred = model(x_batch, adj_tensor)
        all_preds.append(pred.cpu().numpy())
        all_actuals.append(y_batch.numpy())

preds = np.concatenate(all_preds, axis=0)
actuals = np.concatenate(all_actuals, axis=0)

# De-normalize
preds_real   = preds * stds + means
actuals_real = actuals * stds + means

for i, feat in enumerate(FEATURES):
    mae = np.abs(preds_real[:, :, i] - actuals_real[:, :, i]).mean()
    print(f"  MAE ({feat}): {mae:.3f}")


# ── 7. Plot results ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("STGCN Traffic Prediction — Vadodara Road Network (80K rows)",
             fontsize=14, fontweight="bold")

# (a) Training curve
axes[0, 0].plot(train_losses, label="Train Loss", color="#3B8BD4", linewidth=2)
axes[0, 0].plot(test_losses, label="Test Loss", color="#E24B4A", linewidth=2)
axes[0, 0].set_title("Training & Test Loss", fontweight="bold")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# (b) Speed prediction for a congestion-prone road (Jetalpur Road)
jetalpur_idx = edge_to_idx.get("8556786766_8659203468", 0)
pred_speed = preds_real[:, jetalpur_idx, 0]
actual_speed = actuals_real[:, jetalpur_idx, 0]
axes[0, 1].plot(actual_speed[:80], label="Actual", color="#1D9E75", linewidth=2)
axes[0, 1].plot(pred_speed[:80], label="Predicted", color="#E24B4A",
                linewidth=2, linestyle="--")
axes[0, 1].set_title("Speed — Jetalpur Road", fontweight="bold")
axes[0, 1].set_xlabel("Time Step")
axes[0, 1].set_ylabel("Speed (km/h)")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# (c) Scatter: predicted vs actual speed (all roads)
pred_flat = preds_real[:, :, 0].flatten()
actual_flat = actuals_real[:, :, 0].flatten()
axes[1, 0].scatter(actual_flat, pred_flat, alpha=0.1, s=3, color="#3B8BD4")
max_val = max(actual_flat.max(), pred_flat.max())
axes[1, 0].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect")
axes[1, 0].set_title("Predicted vs Actual Speed (All Roads)", fontweight="bold")
axes[1, 0].set_xlabel("Actual Speed (km/h)")
axes[1, 0].set_ylabel("Predicted Speed (km/h)")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# (d) Per-road MAE
road_maes = []
road_labels = []
for eid in edges:
    idx = edge_to_idx[eid]
    mae = np.abs(preds_real[:, idx, 0] - actuals_real[:, idx, 0]).mean()
    road_maes.append(mae)
    road_labels.append(road_names.get(eid, eid)[:12])

colors = ["#E24B4A" if m > np.mean(road_maes) else "#1D9E75" for m in road_maes]
axes[1, 1].barh(road_labels, road_maes, color=colors)
axes[1, 1].set_title("Speed MAE by Road", fontweight="bold")
axes[1, 1].set_xlabel("MAE (km/h)")
axes[1, 1].axvline(np.mean(road_maes), color="#666", linestyle="--", label=f"Avg: {np.mean(road_maes):.1f}")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("results/stgcn_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: results/stgcn_evaluation.png")


# ── 8. Save model ────────────────────────────────────────────
torch.save({
    "model_state_dict": model.state_dict(),
    "num_nodes": num_nodes,
    "in_features": NUM_FEATURES,
    "out_features": OUT_FEATURES,
    "time_steps": TIME_WINDOW,
    "hidden_channels": 64,
    "edges": edges,
    "edge_to_idx": edge_to_idx,
    "features": FEATURES,
    "adj_norm": adj_norm,
    "road_names": road_names,
}, "model/stgcn_model.pt")

# Clean up temp best model
if os.path.exists("model/stgcn_best.pt"):
    os.remove("model/stgcn_best.pt")

print("Saved: model/stgcn_model.pt")
print(f"\nSTGCN training complete!")
print(f"  Data: {len(df):,} rows, {num_nodes} roads")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Best test MSE: {best_test_loss:.6f}")
