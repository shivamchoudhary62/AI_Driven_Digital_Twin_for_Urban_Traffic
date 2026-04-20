"""
evaluate_drl.py
───────────────
Evaluates the trained DRL agent against the fixed-time baseline.
Produces detailed comparison metrics and visualization.

Usage:
    python models/evaluate_drl.py
    python models/evaluate_drl.py --algo dqn
"""

import os, sys, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from models.sumo_env import SumoTrafficEnv

os.makedirs("results", exist_ok=True)


def run_episode(env, agent=None, label="Agent"):
    """
    Run a single episode. If agent is None, uses fixed-time (no action).
    Returns metrics dict.
    """
    state, _ = env.reset()
    total_reward = 0
    total_queue = 0
    total_wait = 0
    total_speed = 0
    steps = 0
    done = False

    step_rewards = []
    step_queues = []

    while not done:
        if agent is not None:
            action, _ = agent.predict(state, deterministic=True)
        else:
            action = 0  # fixed-time: don't change phase

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_rewards.append(reward)

        # Collect queue info
        queue = sum(state[:len(env._incoming_lanes)])  # first N values are queues
        step_queues.append(queue)
        total_queue += queue
        total_wait += info.get("total_waiting", 0)
        steps += 1
        done = terminated or truncated

    avg_queue = total_queue / max(steps, 1)
    avg_reward = total_reward / max(steps, 1)

    print(f"  {label:20s} | Reward: {total_reward:8.1f} | "
          f"Avg Queue: {avg_queue:6.2f} | Steps: {steps}")

    return {
        "label": label,
        "total_reward": total_reward,
        "avg_queue": avg_queue,
        "avg_reward": avg_reward,
        "steps": steps,
        "step_rewards": step_rewards,
        "step_queues": step_queues,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DRL agents")
    parser.add_argument("--algo", type=str, default="both",
                        choices=["dqn", "ppo", "both"])
    args = parser.parse_args()

    env_kwargs = {
        "sim_steps": 3600,
        "delta_time": 5,
        "tl_id": "natubhai",
    }

    results = []

    # ── 1. Run baseline ──────────────────────────────────────
    print("=" * 65)
    print("  EVALUATING: Fixed-Time Baseline vs DRL Agents")
    print("=" * 65)

    env = SumoTrafficEnv(**env_kwargs)
    baseline = run_episode(env, agent=None, label="Fixed-Time Baseline")
    results.append(baseline)
    env.close()

    # ── 2. Run DRL agents ────────────────────────────────────
    algos = {"dqn": DQN, "ppo": PPO}
    if args.algo != "both":
        algos = {args.algo: algos[args.algo]}

    for algo_name, AlgoClass in algos.items():
        model_path = f"model/drl_{algo_name}_agent.zip"
        if not os.path.exists(model_path):
            print(f"  ⚠ {model_path} not found, skipping {algo_name.upper()}")
            continue

        env = SumoTrafficEnv(**env_kwargs)
        agent = AlgoClass.load(model_path)
        result = run_episode(env, agent=agent, label=f"{algo_name.upper()} Agent")
        results.append(result)
        env.close()

    if len(results) < 2:
        print("\n  No trained agents found. Run train_drl.py first.")
        return

    # ── 3. Comparison table ──────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'Metric':<25} ", end="")
    for r in results:
        print(f"  {r['label']:>18}", end="")
    print()
    print("-" * 65)

    metrics = [
        ("Total Reward", "total_reward", True),
        ("Avg Queue Length", "avg_queue", False),
        ("Avg Reward/Step", "avg_reward", True),
    ]

    for label, key, higher_better in metrics:
        print(f"  {label:<25} ", end="")
        for r in results:
            val = r[key]
            print(f"  {val:>18.2f}", end="")
        print()

    # Improvement %
    print()
    for r in results[1:]:
        base_reward = results[0]["total_reward"]
        imp = ((r["total_reward"] - base_reward) / abs(base_reward)) * 100
        queue_imp = ((results[0]["avg_queue"] - r["avg_queue"]) /
                     max(results[0]["avg_queue"], 0.01)) * 100
        print(f"  {r['label']} vs Baseline:")
        print(f"    Reward improvement : {imp:+.1f}%")
        print(f"    Queue reduction    : {queue_imp:+.1f}%")

    # ── 4. Visualization ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DRL Traffic Light Control — Evaluation Results",
                 fontsize=14, fontweight="bold")

    colors = {
        "Fixed-Time Baseline": "#E24B4A",
        "DQN Agent": "#3B8BD4",
        "PPO Agent": "#1D9E75",
    }

    # (a) Reward comparison bar chart
    labels = [r["label"] for r in results]
    rewards = [r["total_reward"] for r in results]
    bar_colors = [colors.get(l, "#999") for l in labels]
    axes[0].bar(labels, rewards, color=bar_colors, edgecolor="none", width=0.5)
    axes[0].set_title("Total Episode Reward", fontweight="bold")
    axes[0].set_ylabel("Reward (higher = better)")
    for i, (l, v) in enumerate(zip(labels, rewards)):
        axes[0].text(i, v + abs(v)*0.02, f"{v:.0f}",
                     ha="center", fontsize=10, fontweight="bold")

    # (b) Queue length over time
    for r in results:
        c = colors.get(r["label"], "#999")
        # Smooth
        qs = r["step_queues"]
        window = max(3, len(qs) // 30)
        smoothed = np.convolve(qs, np.ones(window)/window, mode="valid")
        axes[1].plot(smoothed, label=r["label"], color=c, linewidth=2)
    axes[1].set_title("Queue Length Over Time", fontweight="bold")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Total Queue")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # (c) Average queue comparison
    queues = [r["avg_queue"] for r in results]
    axes[2].bar(labels, queues, color=bar_colors, edgecolor="none", width=0.5)
    axes[2].set_title("Average Queue Length", fontweight="bold")
    axes[2].set_ylabel("Avg Queue (lower = better)")
    for i, (l, v) in enumerate(zip(labels, queues)):
        axes[2].text(i, v + 0.05, f"{v:.2f}",
                     ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/drl_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: results/drl_comparison.png")


if __name__ == "__main__":
    main()
