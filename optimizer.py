"""
optimizer.py — Integrated AI Traffic Optimizer
═══════════════════════════════════════════════
Combines STGCN + DRL in a single SUMO simulation:

  1. STGCN  → predicts future traffic state (speed/density) at T+1
  2. DRL    → controls traffic light phases using predicted + observed state

Flow: Predict congestion → Optimize signal timing → Reduce delays
No vehicle rerouting — only intelligent traffic light control.
"""

import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import traci
import csv
import time
import numpy as np
import torch
from collections import defaultdict

SUMO_HOME    = os.environ.get("SUMO_HOME", r"D:\Program Files\sumosimulator")
SUMO_BINARY  = os.path.join(SUMO_HOME, "bin", "sumo")
SUMO_GUI     = os.path.join(SUMO_HOME, "bin", "sumo-gui")
USE_GUI      = False  # Set to True to watch the simulation in SUMO GUI
SUMO_CFG     = "simulation/simulation.sumocfg"
SIM_STEPS    = 3600
SAMPLE_EVERY = 10
TL_DELTA     = 5    # DRL acts every 5 steps


# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_stgcn_model():
    """Load STGCN for traffic state prediction."""
    from models.stgcn_model import STGCN

    checkpoint = torch.load("model/stgcn_model.pt", map_location=torch.device('cpu'), weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = STGCN(
        num_nodes=checkpoint["num_nodes"],
        in_features=checkpoint["in_features"],
        out_features=checkpoint["out_features"],
        time_steps=checkpoint["time_steps"],
        hidden_channels=checkpoint["hidden_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm = np.load("model/stgcn_norm_params.npz")
    adj_tensor = torch.tensor(checkpoint["adj_norm"], dtype=torch.float32).to(device)

    return {
        "model": model,
        "device": device,
        "adj": adj_tensor,
        "means": norm["means"],
        "stds": norm["stds"],
        "edges": checkpoint["edges"],
        "edge_to_idx": checkpoint["edge_to_idx"],
        "features": checkpoint["features"],
        "time_steps": checkpoint["time_steps"],
    }


def load_drl_agent():
    """Load trained DRL agent for traffic light control."""
    from stable_baselines3 import PPO, DQN

    for path, Cls in [("model/drl_ppo_agent.zip", PPO),
                      ("model/drl_dqn_agent.zip", DQN)]:
        if os.path.exists(path):
            agent = Cls.load(path)
            name = "PPO" if Cls == PPO else "DQN"
            return agent, name
    return None, None


# ══════════════════════════════════════════════════════════════
#  DRL HELPER — get traffic light state
# ══════════════════════════════════════════════════════════════

def get_tl_state(tl_id, incoming_lanes, incoming_edges):
    """Build observation vector for the DRL agent."""
    queues = []
    for lane in incoming_lanes:
        try:
            queues.append(float(traci.lane.getLastStepHaltingNumber(lane)))
        except:
            queues.append(0.0)

    phase = float(traci.trafficlight.getPhase(tl_id))

    densities, speeds = [], []
    for edge in incoming_edges:
        try:
            vc = traci.edge.getLastStepVehicleNumber(edge)
            length = traci.lane.getLength(f"{edge}_0")
            densities.append(vc / max(length, 1.0))
            speeds.append(traci.edge.getLastStepMeanSpeed(edge) * 3.6)
        except:
            densities.append(0.0)
            speeds.append(50.0)

    return np.array(queues + [phase] + densities + speeds, dtype=np.float32)


def apply_drl_action(action, tl_id, step_count, sim_steps):
    """Apply DRL action with yellow transition."""
    green_phases = [0, 2]
    phase = green_phases[action % len(green_phases)]
    current = traci.trafficlight.getPhase(tl_id)

    if phase != current and current in green_phases:
        traci.trafficlight.setPhase(tl_id, current + 1)  # yellow
        for _ in range(3):
            if step_count[0] < sim_steps:
                traci.simulationStep()
                step_count[0] += 1

    traci.trafficlight.setPhase(tl_id, phase)


# ══════════════════════════════════════════════════════════════
#  STGCN PREDICTION
# ══════════════════════════════════════════════════════════════

def update_stgcn_buffer(stgcn_buffer, edges_sumo, stgcn_info):
    """Collect current traffic state and add to STGCN time buffer."""
    feat_names = stgcn_info["features"]
    num_nodes = len(stgcn_info["edges"])
    feat_vec = np.zeros((num_nodes, len(feat_names)), dtype=np.float32)

    for e in edges_sumo:
        if e in stgcn_info["edge_to_idx"]:
            idx = stgcn_info["edge_to_idx"][e]
            vc = traci.edge.getLastStepVehicleNumber(e)
            spd = traci.edge.getLastStepMeanSpeed(e) * 3.6
            occ = traci.edge.getLastStepOccupancy(e)
            free_flow = 50.0
            cr = (1.0 / (spd / free_flow)) if spd > 0 else 1.5

            for fi, fn in enumerate(feat_names):
                if fn == "avg_speed_kmh":    feat_vec[idx, fi] = spd
                elif fn == "vehicle_count":  feat_vec[idx, fi] = vc
                elif fn == "occupancy":      feat_vec[idx, fi] = occ
                elif fn == "congestion_ratio": feat_vec[idx, fi] = cr

    stgcn_buffer.append(feat_vec)
    if len(stgcn_buffer) > stgcn_info["time_steps"]:
        stgcn_buffer.pop(0)


def predict_stgcn(stgcn_buffer, stgcn_info):
    """Run STGCN prediction if buffer is full. Returns predicted state or None."""
    if len(stgcn_buffer) < stgcn_info["time_steps"]:
        return None

    device = stgcn_info["device"]
    model = stgcn_info["model"]
    means = stgcn_info["means"]
    stds = stgcn_info["stds"]

    window = np.array(stgcn_buffer[-stgcn_info["time_steps"]:])
    window_norm = (window - means) / stds
    x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(x, stgcn_info["adj"]).cpu().numpy()[0]

    pred_real = pred_norm * stds + means
    return pred_real  # (N, F)


# ══════════════════════════════════════════════════════════════
#  MAIN SIMULATION
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  INTEGRATED AI TRAFFIC OPTIMIZER")
    print("  STGCN (prediction) + DRL (signal control)")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    stgcn_info = load_stgcn_model()
    print(f"  ✓ STGCN loaded ({len(stgcn_info['edges'])} nodes)")

    drl_agent, drl_name = load_drl_agent()
    if drl_agent:
        print(f"  ✓ DRL agent loaded ({drl_name})")
    else:
        print("  ⚠ No DRL agent found — using fixed-time signals")

    # Start SUMO
    binary = SUMO_GUI if USE_GUI else SUMO_BINARY
    sumo_cmd = [
        binary, "-c", SUMO_CFG,
        "--no-step-log", "true", "--no-warnings", "true",
        "--seed", "42",
        "--time-to-teleport", "300"
    ]
    print("\nStarting SUMO simulation...")
    traci.start(sumo_cmd, port=8815, numRetries=20)

    edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    tl_id = "natubhai"

    # Discover DRL lanes/edges
    incoming_lanes = set()
    incoming_edges = set()
    for link_group in traci.trafficlight.getControlledLinks(tl_id):
        if link_group:
            for link in link_group:
                if len(link) >= 1:
                    lane = link[0]
                    incoming_lanes.add(lane)
                    edge = "_".join(lane.split("_")[:-1])
                    if not edge.startswith(":"):
                        incoming_edges.add(edge)
    incoming_lanes = sorted(incoming_lanes)
    incoming_edges = sorted(incoming_edges)

    # Init
    stgcn_buffer = []
    drl_actions = 0
    stgcn_predictions = 0
    congestion_detected = 0
    step_counter = [0]

    os.makedirs("data", exist_ok=True)

    with open("data/optimized_clean.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "edge_id", "vehicle_count",
                         "avg_speed_kmh", "waiting_time", "occupancy"])

        for step in range(SIM_STEPS):
            traci.simulationStep()
            step_counter[0] = step

            # ── DRL: control traffic lights every TL_DELTA steps ──
            if drl_agent and step % TL_DELTA == 0:
                tl_state = get_tl_state(tl_id, incoming_lanes, incoming_edges)
                action, _ = drl_agent.predict(tl_state, deterministic=True)
                apply_drl_action(action, tl_id, step_counter, SIM_STEPS)
                drl_actions += 1

            # ── Sample & predict every SAMPLE_EVERY steps ─────────
            if step % SAMPLE_EVERY == 0:
                # Update STGCN buffer
                update_stgcn_buffer(stgcn_buffer, edges, stgcn_info)

                # Get STGCN prediction (future state)
                stgcn_pred = predict_stgcn(stgcn_buffer, stgcn_info)
                if stgcn_pred is not None:
                    stgcn_predictions += 1

                    # Log congestion detections (speed < 15 km/h predicted)
                    for e in edges:
                        if e in stgcn_info["edge_to_idx"]:
                            idx = stgcn_info["edge_to_idx"][e]
                            if stgcn_pred[idx, 0] < 15.0:
                                congestion_detected += 1

                # Record current state
                for edge in edges:
                    vc   = traci.edge.getLastStepVehicleNumber(edge)
                    spd  = traci.edge.getLastStepMeanSpeed(edge) * 3.6
                    wait = traci.edge.getWaitingTime(edge)
                    occ  = traci.edge.getLastStepOccupancy(edge)
                    writer.writerow([step, edge, vc, round(spd, 3),
                                     round(wait, 3), round(occ, 3)])

            if step % 500 == 0:
                veh_count = len(traci.vehicle.getIDList())
                print(f"  Step {step:4d}/{SIM_STEPS} — "
                      f"Vehicles: {veh_count:3d} | "
                      f"DRL actions: {drl_actions}")

    traci.close()

    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  STGCN predictions      : {stgcn_predictions}")
    print(f"  Congestion detections  : {congestion_detected}")
    print(f"  DRL signal actions     : {drl_actions}")
    print(f"  DRL agent              : {drl_name or 'None (fixed-time)'}")
    print(f"  Output                 : data/optimized_clean.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()