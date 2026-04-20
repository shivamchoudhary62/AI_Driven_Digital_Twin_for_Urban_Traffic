import pandas as pd

b = pd.read_csv("data/baseline_clean.csv")
o = pd.read_csv("data/optimized_clean.csv")

print("=== BASELINE ===")
print(f"Rows: {len(b)}, Steps: {b['step'].nunique()}, Range: {b['step'].min()}-{b['step'].max()}")
print(f"Avg Wait: {b['waiting_time'].mean():.2f}s | Max Wait: {b['waiting_time'].max():.0f}s")
print(f"Avg Speed: {b['avg_speed_kmh'].mean():.1f} km/h | Avg Occ: {b['occupancy'].mean():.4f}")

print("\n=== OPTIMIZED ===")
print(f"Rows: {len(o)}, Steps: {o['step'].nunique()}, Range: {o['step'].min()}-{o['step'].max()}")
print(f"Avg Wait: {o['waiting_time'].mean():.2f}s | Max Wait: {o['waiting_time'].max():.0f}s")
print(f"Avg Speed: {o['avg_speed_kmh'].mean():.1f} km/h | Avg Occ: {o['occupancy'].mean():.4f}")

print("\n=== IMPROVEMENT ===")
wait_imp = (b['waiting_time'].mean() - o['waiting_time'].mean()) / (b['waiting_time'].mean() + 1e-9) * 100
spd_imp = (o['avg_speed_kmh'].mean() - b['avg_speed_kmh'].mean()) / (b['avg_speed_kmh'].mean() + 1e-9) * 100
occ_imp = (b['occupancy'].mean() - o['occupancy'].mean()) / (b['occupancy'].mean() + 1e-9) * 100
print(f"Wait time reduction: {wait_imp:.1f}%")
print(f"Speed improvement:   {spd_imp:.1f}%")
print(f"Occupancy reduction: {occ_imp:.1f}%")
