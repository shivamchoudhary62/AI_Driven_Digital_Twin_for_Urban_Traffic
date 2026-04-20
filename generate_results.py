import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

os.makedirs("results", exist_ok=True)

print("Loading data...")
b = pd.read_csv("data/baseline_clean.csv")
o = pd.read_csv("data/optimized_clean.csv")

# Aggregate per step
b_step = b.groupby("step").agg(
    avg_wait  = ("waiting_time",  "mean"),
    avg_speed = ("avg_speed_kmh", "mean"),
    avg_occ   = ("occupancy",     "mean"),
    avg_vc    = ("vehicle_count", "mean")
).reset_index()

o_step = o.groupby("step").agg(
    avg_wait  = ("waiting_time",  "mean"),
    avg_speed = ("avg_speed_kmh", "mean"),
    avg_occ   = ("occupancy",     "mean"),
    avg_vc    = ("vehicle_count", "mean")
).reset_index()

# ── Color scheme ──────────────────────────────────────────
C_BASE = "#E24B4A"   # red for baseline
C_OPT  = "#1D9E75"   # green for optimized
C_FILL_B = "#E24B4A"
C_FILL_O = "#1D9E75"

# ── Figure 1 — Main 4-panel comparison ───────────────────
fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    "AI-Driven Digital Twin — Traffic Optimization Results\n"
    "Baseline vs AI-Optimized Simulation",
    fontsize=15, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Waiting time over time
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(b_step["step"], b_step["avg_wait"],
         color=C_BASE, linewidth=1.5, label="Baseline", alpha=0.9)
ax1.plot(o_step["step"], o_step["avg_wait"],
         color=C_OPT,  linewidth=1.5, label="AI Optimized", alpha=0.9)
ax1.fill_between(b_step["step"], b_step["avg_wait"],
                 alpha=0.12, color=C_FILL_B)
ax1.fill_between(o_step["step"], o_step["avg_wait"],
                 alpha=0.12, color=C_FILL_O)
ax1.set_title("Average Waiting Time Over Simulation", fontweight="bold")
ax1.set_xlabel("Simulation Step")
ax1.set_ylabel("Waiting Time (seconds)")
ax1.legend()

# Panel 2 — Speed over time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(b_step["step"], b_step["avg_speed"],
         color=C_BASE, linewidth=1.5, label="Baseline", alpha=0.9)
ax2.plot(o_step["step"], o_step["avg_speed"],
         color=C_OPT,  linewidth=1.5, label="AI Optimized", alpha=0.9)
ax2.fill_between(b_step["step"], b_step["avg_speed"],
                 alpha=0.12, color=C_FILL_B)
ax2.fill_between(o_step["step"], o_step["avg_speed"],
                 alpha=0.12, color=C_FILL_O)
ax2.set_title("Average Speed Over Simulation", fontweight="bold")
ax2.set_xlabel("Simulation Step")
ax2.set_ylabel("Speed (km/h)")
ax2.legend()

# Panel 3 — Summary bar chart
ax3 = fig.add_subplot(gs[1, 0])
metrics_labels = [
    "Avg Wait\n(s)",
    "Max Wait\n(s)",
    "Avg\nOccupancy",
]
b_vals = [
    b["waiting_time"].mean(),
    b["waiting_time"].max(),
    b["occupancy"].mean() * 100,
]
o_vals = [
    o["waiting_time"].mean(),
    o["waiting_time"].max(),
    o["occupancy"].mean() * 100,
]

# Normalize each metric to percentage of baseline for fair display
b_norm = [100, 100, 100]
o_norm = [round(o_vals[i] / (b_vals[i] + 1e-9) * 100, 1)
          for i in range(len(b_vals))]

x = np.arange(len(metrics_labels))
bars_b = ax3.bar(x - 0.2, b_norm, 0.38,
                  label="Baseline (100%)", color=C_BASE, alpha=0.85)
bars_o = ax3.bar(x + 0.2, o_norm, 0.38,
                  label="AI Optimized", color=C_OPT, alpha=0.85)

ax3.set_xticks(x)
ax3.set_xticklabels(metrics_labels)
ax3.set_ylabel("% of Baseline")
ax3.set_title("Performance Comparison\n(lower = better)", fontweight="bold")
ax3.legend()
ax3.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

for bar in bars_o:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
             f"{h:.1f}%", ha="center", va="bottom", fontsize=9,
             color="#0F6E56", fontweight="bold")

# Panel 4 — Improvement summary table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")

table_data = [
    ["Metric", "Baseline", "Optimized", "Improvement"],
    ["Avg waiting time",
     f"{b['waiting_time'].mean():.2f}s",
     f"{o['waiting_time'].mean():.2f}s",
     f"↓ {(b['waiting_time'].mean()-o['waiting_time'].mean())/b['waiting_time'].mean()*100:.1f}%"],
    ["Max waiting time",
     f"{b['waiting_time'].max():.0f}s",
     f"{o['waiting_time'].max():.0f}s",
     f"↓ {(b['waiting_time'].max()-o['waiting_time'].max())/b['waiting_time'].max()*100:.1f}%"],
    ["Avg speed",
     f"{b['avg_speed_kmh'].mean():.2f}",
     f"{o['avg_speed_kmh'].mean():.2f}",
     f"↑ {(o['avg_speed_kmh'].mean()-b['avg_speed_kmh'].mean())/b['avg_speed_kmh'].mean()*100:.1f}%"],
    ["Avg occupancy",
     f"{b['occupancy'].mean():.4f}",
     f"{o['occupancy'].mean():.4f}",
     f"↓ {(b['occupancy'].mean()-o['occupancy'].mean())/b['occupancy'].mean()*100:.1f}%"],
]

table = ax4.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0, 0.1, 1, 0.85]
)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Style header row
for j in range(4):
    table[0, j].set_facecolor("#2C2C2A")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Style improvement column green
for i in range(1, len(table_data)):
    table[i, 3].set_facecolor("#E1F5EE")
    table[i, 3].set_text_props(color="#0F6E56", fontweight="bold")

ax4.set_title("Summary Table", fontweight="bold", pad=10)

plt.savefig("results/final_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: results/final_comparison.png")

# ── Figure 2 — Waiting time distribution ─────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Waiting Time Distribution — Baseline vs Optimized",
              fontsize=13, fontweight="bold")

# Histogram
axes2[0].hist(b["waiting_time"], bins=40, color=C_BASE,
              alpha=0.7, label="Baseline", density=True)
axes2[0].hist(o["waiting_time"], bins=40, color=C_OPT,
              alpha=0.7, label="AI Optimized", density=True)
axes2[0].set_xlabel("Waiting Time (seconds)")
axes2[0].set_ylabel("Density")
axes2[0].set_title("Waiting Time Distribution", fontweight="bold")
axes2[0].legend()

# Per-edge waiting time comparison
edge_b = b.groupby("edge_id")["waiting_time"].mean().sort_values(ascending=False)
edge_o = o.groupby("edge_id")["waiting_time"].mean().reindex(edge_b.index)

x = np.arange(len(edge_b))
axes2[1].bar(x - 0.2, edge_b.values, 0.38,
             label="Baseline", color=C_BASE, alpha=0.85)
axes2[1].bar(x + 0.2, edge_o.values, 0.38,
             label="AI Optimized", color=C_OPT, alpha=0.85)
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(edge_b.index, rotation=45, ha="right", fontsize=9)
axes2[1].set_ylabel("Avg Waiting Time (s)")
axes2[1].set_title("Per-Edge Waiting Time", fontweight="bold")
axes2[1].legend()

plt.tight_layout()
plt.savefig("results/waiting_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: results/waiting_distribution.png")

print("\n=== FINAL RESULTS SUMMARY ===")
print(f"Avg waiting time : {b['waiting_time'].mean():.2f}s → {o['waiting_time'].mean():.2f}s  (↓ {(b['waiting_time'].mean()-o['waiting_time'].mean())/b['waiting_time'].mean()*100:.1f}%)")
print(f"Max waiting time : {b['waiting_time'].max():.0f}s → {o['waiting_time'].max():.0f}s  (↓ {(b['waiting_time'].max()-o['waiting_time'].max())/b['waiting_time'].max()*100:.1f}%)")
print(f"Avg speed        : {b['avg_speed_kmh'].mean():.2f} → {o['avg_speed_kmh'].mean():.2f} km/h  (↑ {(o['avg_speed_kmh'].mean()-b['avg_speed_kmh'].mean())/b['avg_speed_kmh'].mean()*100:.1f}%)")
print(f"Avg occupancy    : {b['occupancy'].mean():.4f} → {o['occupancy'].mean():.4f}  (↓ {(b['occupancy'].mean()-o['occupancy'].mean())/b['occupancy'].mean()*100:.1f}%)")
print("\nAll graphs saved to results/")
print("Day 6 complete.")
