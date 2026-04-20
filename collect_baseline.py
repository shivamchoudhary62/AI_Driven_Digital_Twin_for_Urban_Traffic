import traci
import csv
import os
import numpy as np

# ── Absolute paths ────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SUMO_HOME   = os.environ.get("SUMO_HOME", r"D:\Program Files\sumosimulator")
SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo")
SUMO_CFG    = os.path.join(BASE_DIR, "simulation", "simulation.sumocfg")
OUTPUT_CSV  = os.path.join(BASE_DIR, "data", "baseline_clean.csv")
SIM_STEPS   = 3600
SAMPLE_EVERY = 10

print(f"Config path: {SUMO_CFG}")
print(f"Config exists: {os.path.exists(SUMO_CFG)}")

sumo_cmd = [
    SUMO_BINARY,
    "-c", SUMO_CFG,
    "--no-step-log", "true",
    "--no-warnings", "true",
    "--seed", "42"
]

print("Collecting clean baseline...")
traci.start(sumo_cmd, port=8813, numRetries=20)

edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
tls_list = traci.trafficlight.getIDList()
print(f"Edges: {len(edges)}, Traffic lights: {len(tls_list)}")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "step", "edge_id", "vehicle_count",
        "avg_speed_kmh", "waiting_time", "occupancy"
    ])

    for step in range(SIM_STEPS):
        traci.simulationStep()
        if step % SAMPLE_EVERY == 0:
            for edge in edges:
                writer.writerow([
                    step,
                    edge,
                    traci.edge.getLastStepVehicleNumber(edge),
                    round(traci.edge.getLastStepMeanSpeed(edge) * 3.6, 3),
                    round(traci.edge.getWaitingTime(edge), 3),
                    round(traci.edge.getLastStepOccupancy(edge), 3)
                ])
        if step % 500 == 0:
            print(f"  Step {step}/{SIM_STEPS}")

traci.close()
print(f"Baseline saved to {OUTPUT_CSV}")
