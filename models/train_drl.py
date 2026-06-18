"""
train_drl.py
────────────
Trains Deep Reinforcement Learning agents (DQN and PPO) for traffic light
control using the SUMO Gymnasium environment.

Usage:
    python models/train_drl.py                    # default: 300 episodes
    python models/train_drl.py --episodes 500     # custom
    python models/train_drl.py --algo ppo         # PPO only
    python models/train_drl.py --algo dqn         # DQN only

Output:
    model/drl_dqn_agent.zip   — trained DQN policy
    model/drl_ppo_agent.zip   — trained PPO policy
    results/drl_learning_curve.png
"""

import os, sys, argparse, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from models.sumo_env import SumoTrafficEnv

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Reward logging callback ──────────────────────────────────
class RewardLogger(BaseCallback):
    """Logs episode rewards for plotting learning curves."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_reward = 0
        self._current_length = 0

    def _on_step(self):
        self._current_reward += self.locals["rewards"][0]
        self._current_length += 1

        # Check if episode ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            if len(self.episode_rewards) % 10 == 0:
                avg = np.mean(self.episode_rewards[-10:])
                print(f"    Episode {len(self.episode_rewards):4d}  "
                      f"Reward: {self._current_reward:8.1f}  "
                      f"Avg(10): {avg:8.1f}")
            self._current_reward = 0
            self._current_length = 0
        return True


def train_agent(algo_name, total_timesteps, env_kwargs):
    """Train a single RL agent."""
    print(f"\n{'='*55}")
    print(f"  Training {algo_name.upper()} Agent")
    print(f"{'='*55}")

    env = Monitor(SumoTrafficEnv(**env_kwargs))
    callback = RewardLogger()

    if algo_name == "dqn":
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=500,
            batch_size=64,
            gamma=0.99,
            target_update_interval=500,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=0,
            device="auto",
        )
    elif algo_name == "ppo":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            verbose=0,
            device="auto",
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    print(f"  Total timesteps: {total_timesteps:,}")
    start = time.time()

    model.learn(total_timesteps=total_timesteps, callback=callback)

    elapsed = time.time() - start
    print(f"  Training time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    save_path = f"model/drl_{algo_name}_agent"
    model.save(save_path)
    print(f"  Saved: {save_path}.zip")

    env.close()

    return callback.episode_rewards, callback.episode_lengths


def run_baseline_episode(env_kwargs):
    """Run one episode with fixed-time traffic lights (no RL)."""
    print("\nRunning fixed-time baseline...")
    env = SumoTrafficEnv(**env_kwargs)
    state, _ = env.reset()

    total_reward = 0
    steps = 0
    done = False

    while not done:
        # Action 0 = keep default phase cycling (don't interfere)
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    env.close()
    print(f"  Baseline reward: {total_reward:.1f} over {steps} steps")
    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Train DRL traffic agents")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Approximate number of episodes")
    parser.add_argument("--algo", type=str, default="both",
                        choices=["dqn", "ppo", "both"],
                        help="Algorithm to train")
    args = parser.parse_args()

    # Each episode ~720 steps (3600 sim / 5 delta), so total_timesteps ≈ episodes * 720
    steps_per_episode = 720
    total_timesteps = args.episodes * steps_per_episode

    env_kwargs = {
        "sim_steps": 3600,
        "delta_time": 5,
        "tl_id": "natubhai",
    }

    results = {}

    if args.algo in ("dqn", "both"):
        rewards_dqn, _ = train_agent("dqn", total_timesteps, env_kwargs)
        results["DQN"] = rewards_dqn

    if args.algo in ("ppo", "both"):
        rewards_ppo, _ = train_agent("ppo", total_timesteps, env_kwargs)
        results["PPO"] = rewards_ppo

    # Baseline
    baseline_reward = run_baseline_episode(env_kwargs)

    # ── Plot learning curves ──────────────────────────────────
    colors = {"DQN": "#3B8BD4", "PPO": "#1D9E75"}
    num_algos = len(results)
    num_cols = num_algos + 1  # individual panels + combined

    # Compute global Y-axis range from agent rewards (exclude baseline outlier)
    all_rewards = []
    for rewards in results.values():
        all_rewards.extend(rewards)
    y_min = min(all_rewards) if all_rewards else -500
    y_max = max(all_rewards) if all_rewards else 100
    y_pad = max((y_max - y_min) * 0.15, 20)

    fig, axes = plt.subplots(1, num_cols, figsize=(7 * num_cols, 6))
    fig.suptitle("DRL Training — Learning Curves",
                 fontsize=16, fontweight="bold", y=1.02)

    if num_cols == 1:
        axes = [axes]

    def _add_baseline_annotation(ax):
        """Show baseline as text annotation instead of axhline to avoid scale distortion."""
        ax.annotate(
            f"⬇ Fixed-time baseline: {baseline_reward:.0f}",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=9, color="#E24B4A", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0",
                      edgecolor="#E24B4A", alpha=0.9),
        )

    # ── Individual subplot per algorithm ──
    for i, (name, rewards) in enumerate(results.items()):
        ax = axes[i]
        window = max(5, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

        ax.plot(rewards, alpha=0.25, color=colors.get(name, "#333"),
                linewidth=0.8, label="Raw")
        ax.plot(smoothed, color=colors.get(name, "#333"),
                linewidth=2.5, label=f"{name} (smoothed)")

        final_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        ax.axhline(y=final_avg, color=colors.get(name, "#333"),
                   linestyle=":", linewidth=1.2, alpha=0.7,
                   label=f"Final avg: {final_avg:.1f}")
        ax.axhline(y=0, color="#999", linestyle="-", linewidth=0.5, alpha=0.5)

        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_title(f"{name} Learning Curve", fontweight="bold", fontsize=13)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        _add_baseline_annotation(ax)

    # ── Combined overlay (rightmost panel) ──
    ax_comb = axes[-1]
    for name, rewards in results.items():
        window = max(5, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax_comb.plot(smoothed, label=f"{name} (smoothed)",
                     color=colors.get(name, "#333"), linewidth=2.5)
        ax_comb.plot(rewards, alpha=0.15, color=colors.get(name, "#333"),
                     linewidth=0.5)

    ax_comb.axhline(y=0, color="#999", linestyle="-", linewidth=0.5, alpha=0.5)
    ax_comb.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_comb.set_title("Combined Comparison", fontweight="bold", fontsize=13)
    ax_comb.set_xlabel("Episode", fontsize=11)
    ax_comb.set_ylabel("Episode Reward", fontsize=11)
    ax_comb.legend(fontsize=9, loc="lower right")
    ax_comb.grid(alpha=0.3)
    _add_baseline_annotation(ax_comb)

    plt.tight_layout()
    plt.savefig("results/drl_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/drl_learning_curve.png")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  DRL TRAINING COMPLETE")
    print(f"{'='*55}")
    print(f"  Baseline reward     : {baseline_reward:.1f}")
    for name, rewards in results.items():
        final_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        best = max(rewards)
        print(f"  {name} final avg(10) : {final_avg:.1f}")
        print(f"  {name} best episode  : {best:.1f}")
        improvement = ((final_avg - baseline_reward) / abs(baseline_reward)) * 100
        print(f"  {name} improvement   : {improvement:+.1f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
