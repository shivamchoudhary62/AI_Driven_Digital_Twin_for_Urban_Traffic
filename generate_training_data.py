"""
generate_training_data.py
─────────────────────────
Reads the 360 real data points from real_traffic_data.csv, extracts
per-road traffic profiles, and generates ~50,000+ synthetic but realistic
training rows spanning multiple weeks / all hours / all days.

Target label: is_congested_next  (will this road be congested in the NEXT
time window?)
"""

import os, random, math
import numpy as np
import pandas as pd
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── reproducibility ───────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── 1. Read real data & extract per-road profiles ─────────────
print("Reading real_traffic_data.csv …")
raw = pd.read_csv("data/real_traffic_data.csv")

# Road profiles from the real observations
road_profiles = {}
for edge_id, grp in raw.groupby("edge_id"):
    name = grp["edge_name"].iloc[0]
    road_profiles[edge_id] = {
        "edge_name": name,
        "length_m": grp["length_m"].iloc[0],
        # speed stats
        "speed_mean": grp["avg_speed_kmh"].mean(),
        "speed_std": grp["avg_speed_kmh"].std() + 0.5,  # min noise
        "speed_min": max(grp["avg_speed_kmh"].min() - 5, 3),
        "speed_max": grp["avg_speed_kmh"].max() + 8,
        # vehicle count stats
        "vc_mean": grp["vehicle_count"].mean(),
        "vc_std": max(grp["vehicle_count"].std(), 0.8),
        "vc_min": max(grp["vehicle_count"].min() - 1, 0),
        "vc_max": grp["vehicle_count"].max() + 4,
        # congestion ratio
        "cr_mean": grp["congestion_ratio"].mean(),
        "cr_std": max(grp["congestion_ratio"].std(), 0.04),
        # waiting time
        "wt_mean": grp["waiting_time"].mean(),
        "wt_max": grp["waiting_time"].max() + 5,
        # occupancy
        "occ_mean": grp["occupancy"].mean(),
        "occ_max": max(grp["occupancy"].max(), 0.02),
    }

print(f"  Found {len(road_profiles)} road profiles")

# ── 2. Time-of-day traffic multipliers ────────────────────────
#   Returns a multiplier [0..1] for "how busy" the hour is.
#   Peak hours get multiplier ≈ 1, night ≈ 0.1

def traffic_multiplier(hour: int, minute: int, is_weekend: bool) -> float:
    """Realistic Indian city traffic curve."""
    t = hour + minute / 60.0

    if is_weekend:
        # Weekend: gentler peaks, shifted later
        if 10 <= t < 13:      return 0.70 + 0.15 * math.sin((t - 10) / 3 * math.pi)
        elif 16 <= t < 20:    return 0.65 + 0.15 * math.sin((t - 16) / 4 * math.pi)
        elif 0 <= t < 6:      return 0.08 + 0.04 * (t / 6)
        elif 6 <= t < 10:     return 0.25 + 0.35 * ((t - 6) / 4)
        elif 13 <= t < 16:    return 0.45
        elif 20 <= t < 23:    return 0.35 - 0.10 * ((t - 20) / 3)
        else:                 return 0.10
    else:
        # Weekday peaks
        if 8 <= t < 10:       return 0.80 + 0.20 * math.sin((t - 8) / 2 * math.pi)
        elif 17 <= t < 20:    return 0.85 + 0.15 * math.sin((t - 17) / 3 * math.pi)
        elif 0 <= t < 5:      return 0.05 + 0.03 * (t / 5)
        elif 5 <= t < 8:      return 0.15 + 0.55 * ((t - 5) / 3)
        elif 10 <= t < 12:    return 0.60 + 0.10 * math.sin((t - 10) / 2 * math.pi)
        elif 12 <= t < 14:    return 0.70  # lunch hour
        elif 14 <= t < 17:    return 0.55 + 0.15 * ((t - 14) / 3)
        elif 20 <= t < 23:    return 0.40 - 0.15 * ((t - 20) / 3)
        else:                 return 0.08


# ── 3. Generate synthetic data ────────────────────────────────
print("Generating synthetic training data …")

NUM_WEEKS = 4          # 4 weeks of data
INTERVAL_MIN = 5       # sample every 5 minutes
SLOTS_PER_DAY = 24 * 60 // INTERVAL_MIN   # 288

rows = []
edges = list(road_profiles.keys())

for week in range(NUM_WEEKS):
    for day in range(7):         # 0=Mon … 6=Sun
        is_weekend = day >= 5
        for slot in range(SLOTS_PER_DAY):
            hour   = (slot * INTERVAL_MIN) // 60
            minute = (slot * INTERVAL_MIN) % 60
            mult   = traffic_multiplier(hour, minute, is_weekend)

            for edge_id in edges:
                p = road_profiles[edge_id]

                # ── vehicle count ──
                base_vc = p["vc_mean"] * (0.3 + 1.4 * mult)
                vc = int(np.clip(
                    np.random.normal(base_vc, p["vc_std"] * (0.5 + mult)),
                    p["vc_min"], p["vc_max"] + int(6 * mult)
                ))

                # ── average speed (inversely related to traffic) ──
                speed_factor = 1.0 - 0.55 * mult  # heavy traffic → slower
                base_speed = p["speed_mean"] * speed_factor
                noise = np.random.normal(0, p["speed_std"] * 0.6)
                avg_speed = np.clip(base_speed + noise, p["speed_min"], p["speed_max"])

                # ── congestion ratio ──
                cr_base = p["cr_mean"] * (0.6 + 0.8 * mult)
                cr = np.clip(
                    np.random.normal(cr_base, p["cr_std"] * 1.2),
                    0.3, 2.0
                )

                # ── waiting time (exponential during peaks) ──
                if mult > 0.7:
                    wt = np.clip(np.random.exponential(p["wt_mean"] * mult * 3), 0, 60)
                elif mult > 0.4:
                    wt = np.clip(np.random.exponential(p["wt_mean"] * mult * 1.5), 0, 30)
                else:
                    wt = np.clip(np.random.exponential(max(p["wt_mean"] * 0.3, 0.1)), 0, 5)

                # ── occupancy ──
                occ_base = p["occ_mean"] * (0.2 + 1.5 * mult)
                occ = np.clip(np.random.normal(occ_base, 0.005), 0, 0.15)

                # ── derived features ──
                free_flow_speed = p["speed_max"]
                speed_ratio = avg_speed / free_flow_speed if free_flow_speed > 0 else 1.0
                density = vc / p["length_m"] if p["length_m"] > 0 else 0

                rows.append({
                    "week":              week,
                    "day_of_week":       day,
                    "hour":              hour,
                    "minute":            minute,
                    "slot":              slot,
                    "edge_id":           edge_id,
                    "length_m":          p["length_m"],
                    "vehicle_count":     vc,
                    "avg_speed_kmh":     round(avg_speed, 2),
                    "waiting_time":      round(wt, 2),
                    "occupancy":         round(occ, 6),
                    "congestion_ratio":  round(cr, 4),
                    "speed_ratio":       round(speed_ratio, 4),
                    "density":           round(density, 6),
                })

df = pd.DataFrame(rows)
print(f"  Generated {len(df)} raw rows")

# ── 4. Define "congested" for the CURRENT row ─────────────────
def is_congested(row):
    if row["congestion_ratio"] > 1.1:
        return 1
    if row["avg_speed_kmh"] < 15:
        return 1
    if row["waiting_time"] > 10:
        return 1
    return 0

df["is_congested"] = df.apply(is_congested, axis=1)

# ── 5. Compute target: is_congested_NEXT ──────────────────────
#   For each road, look at the NEXT time slot and check if it is congested.
#   This lets the model PREDICT upcoming congestion.

print("Computing future congestion labels …")
df = df.sort_values(["edge_id", "week", "day_of_week", "slot"]).reset_index(drop=True)

df["is_congested_next"] = (
    df.groupby(["edge_id", "week", "day_of_week"])["is_congested"]
      .shift(-1)            # look one step ahead
      .fillna(0)            # last slot of day → assume no congestion
      .astype(int)
)

# ── 6. Drop helper columns, keep only training-ready ones ─────
keep_cols = [
    "hour", "minute", "day_of_week",
    "edge_id", "length_m",
    "vehicle_count", "avg_speed_kmh", "waiting_time",
    "occupancy", "congestion_ratio",
    "speed_ratio", "density",
    "is_congested_next"
]
df_final = df[keep_cols].copy()

# ── 7. Write output ──────────────────────────────────────────
out_path = "data/real_traffic_data.csv"
backup_path = "data/real_traffic_data_original.csv"

# Backup original
if not os.path.exists(backup_path):
    import shutil
    shutil.copy2(out_path, backup_path)
    print(f"  Backed up original → {backup_path}")

df_final.to_csv(out_path, index=False)

# ── 8. Summary stats ─────────────────────────────────────────
total = len(df_final)
positive = df_final["is_congested_next"].sum()
negative = total - positive

print(f"\n{'='*55}")
print(f"  DATASET GENERATED SUCCESSFULLY")
print(f"{'='*55}")
print(f"  Output         : {out_path}")
print(f"  Total rows     : {total:,}")
print(f"  Congested-next : {positive:,}  ({100*positive/total:.1f}%)")
print(f"  Normal-next    : {negative:,}  ({100*negative/total:.1f}%)")
print(f"  Columns        : {list(df_final.columns)}")
print(f"  Roads          : {df_final['edge_id'].nunique()}")
print(f"  Days simulated : {NUM_WEEKS * 7}")
print(f"{'='*55}")
