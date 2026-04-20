"""
AI-Driven Digital Twin for Urban Traffic Optimization
=====================================================
Run the entire pipeline with: python main.py

Pipeline:
  Step 1 — Collect baseline simulation data
  Step 2 — Train STGCN traffic predictor (GNN)
  Step 3 — Train DRL traffic light agent (RL)
  Step 4 — Run DRL-optimized simulation & compare
  Step 5 — Generate final comparison results
"""

import os
import sys
import time
import subprocess

# ── Config ────────────────────────────────────────────────
PYTHON  = sys.executable
STEPS   = [
    ("Step 1 — Collecting baseline traffic data",
     "collect_baseline.py"),
    ("Step 2 — Training STGCN traffic predictor (GNN)",
     "models/train_stgcn.py"),
    ("Step 3 — Training DRL traffic light agent",
     "models/train_drl.py"),
    ("Step 4 — Evaluating DRL vs baseline",
     "models/evaluate_drl.py"),
    ("Step 5 — Generating final comparison results",
     "generate_results.py"),
]
# ─────────────────────────────────────────────────────────

def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def run_step(label, script):
    banner(label)
    start = time.time()
    result = subprocess.run(
        [PYTHON, script],
        capture_output=False
    )
    elapsed = round(time.time() - start, 1)
    if result.returncode == 0:
        print(f"\n  ✓ Completed in {elapsed}s")
        return True
    else:
        print(f"\n  ✗ FAILED after {elapsed}s")
        return False

def kill_sumo():
    os.system('taskkill /F /IM sumo.exe >nul 2>&1')
    os.system('taskkill /F /IM sumo-gui.exe >nul 2>&1')
    time.sleep(2)

if __name__ == "__main__":
    total_start = time.time()

    print("""
    +======================================================+
    |   AI-Driven Digital Twin                              |
    |   Urban Traffic Optimization System                   |
    |                                                       |
    |   STGCN Prediction + DRL Optimization                 |
    +======================================================+
    """)

    for label, script in STEPS:
        kill_sumo()
        success = run_step(label, script)
        if not success:
            print(f"\nPipeline stopped at: {script}")
            print("Fix the error above and re-run main.py")
            sys.exit(1)

    total = round(time.time() - total_start, 1)

    banner("Pipeline Complete")
    print(f"""
  Total time        : {total}s

  Output files:
    data/baseline_clean.csv              — baseline simulation data
    model/stgcn_model.pt                 — trained STGCN predictor
    model/drl_dqn_agent.zip              — trained DQN agent
    model/drl_ppo_agent.zip              — trained PPO agent
    results/stgcn_evaluation.png         — STGCN prediction accuracy
    results/drl_learning_curve.png       — DRL training progress
    results/drl_comparison.png           — DRL vs baseline comparison
    results/model_evaluation.png         — RF model performance

  Individual scripts:
    python models/train_stgcn.py         — retrain STGCN
    python models/train_drl.py --algo ppo --episodes 500  — retrain DRL
    python models/evaluate_drl.py        — re-evaluate
    """)
