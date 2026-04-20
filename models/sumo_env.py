"""
sumo_env.py
───────────
Custom Gymnasium environment wrapping SUMO via TraCI for
Deep Reinforcement Learning traffic light control.

Controls Natubhai Circle intersection (central hub with 11 links).

State:  queue lengths on incoming lanes + current phase + density
Action: select traffic light phase (0-3)
Reward: negative sum of queue lengths (minimize queues)
"""

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

SUMO_HOME   = os.environ.get("SUMO_HOME", r"D:\Program Files\sumosimulator")
SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo")
SUMO_CFG    = "simulation/simulation.sumocfg"


class SumoTrafficEnv(gym.Env):
    """
    RL Environment for traffic light control at Natubhai Circle.

    MDP formulation:
        State:  [queue_lane_1, ..., queue_lane_n, current_phase,
                 density_edge_1, ..., density_edge_m, avg_speed_1, ..., avg_speed_m]
        Action: {0, 1} — select green phase (NS or EW)
        Reward: R_t = -Σ queue_lengths  (encourage flow)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_steps=3600, delta_time=5, tl_id="natubhai",
                 port=None, gui=False):
        super().__init__()

        self.sim_steps = sim_steps
        self.delta_time = delta_time  # steps between actions
        self.tl_id = tl_id
        self.gui = gui

        self._sumo_running = False
        self._step_count = 0
        self._num_phases = 4

        # Probe SUMO once to get exact observation shape
        self._incoming_lanes = []
        self._incoming_edges = []
        self._probe_network()

        # Now we know exact sizes
        obs_size = len(self._incoming_lanes) + 1 + \
                   len(self._incoming_edges) + len(self._incoming_edges)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: choose one of 2 green phases (NS or EW)
        self.action_space = spaces.Discrete(2)

    def _probe_network(self):
        """Start SUMO briefly to discover lane/edge topology."""
        port = np.random.randint(9000, 12000)
        sumo_cmd = [
            SUMO_BINARY, "-c", SUMO_CFG,
            "--no-step-log", "true", "--no-warnings", "true",
        ]
        traci.start(sumo_cmd, port=port, numRetries=30)
        traci.simulationStep()
        self._discover_incoming()
        traci.close()
        self._sumo_running = False

    def _discover_incoming(self):
        """Discover lanes and edges feeding into the controlled intersection."""
        controlled = traci.trafficlight.getControlledLinks(self.tl_id)
        lanes = set()
        edges = set()
        for link_group in controlled:
            if link_group:
                for link in link_group:
                    if len(link) >= 1:
                        lane = link[0]
                        lanes.add(lane)
                        edge = "_".join(lane.split("_")[:-1])
                        if not edge.startswith(":"):
                            edges.add(edge)

        self._incoming_lanes = sorted(lanes)
        self._incoming_edges = sorted(edges)

    def _start_sumo(self):
        """Start SUMO simulation."""
        if self._sumo_running:
            try:
                traci.close()
            except:
                pass

        binary = SUMO_BINARY
        if self.gui:
            binary = os.path.join(SUMO_HOME, "bin", "sumo-gui")

        sumo_cmd = [
            binary, "-c", SUMO_CFG,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--seed", str(np.random.randint(0, 100000)),
            "--time-to-teleport", "-1",
        ]

        port = np.random.randint(9000, 12000)
        traci.start(sumo_cmd, port=port, numRetries=30)
        self._sumo_running = True

    def _get_state(self):
        """Build the observation vector."""
        queues = []
        for lane in self._incoming_lanes:
            try:
                q = traci.lane.getLastStepHaltingNumber(lane)
                queues.append(float(q))
            except:
                queues.append(0.0)

        phase = float(traci.trafficlight.getPhase(self.tl_id))

        densities = []
        speeds = []
        for edge in self._incoming_edges:
            try:
                vc = traci.edge.getLastStepVehicleNumber(edge)
                length = traci.lane.getLength(f"{edge}_0")
                densities.append(vc / max(length, 1.0))
                speeds.append(traci.edge.getLastStepMeanSpeed(edge) * 3.6)
            except:
                densities.append(0.0)
                speeds.append(50.0)

        state = np.array(queues + [phase] + densities + speeds, dtype=np.float32)
        return state

    def _get_reward(self):
        """
        Reward = -Σ queue_lengths across all incoming lanes.
        Lower queues → higher reward.
        """
        total_queue = 0
        for lane in self._incoming_lanes:
            try:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
            except:
                pass

        throughput = 0
        for edge in self._incoming_edges:
            try:
                throughput += traci.edge.getLastStepVehicleNumber(edge)
            except:
                pass

        reward = -total_queue + 0.1 * throughput
        return reward

    def _apply_action(self, action):
        """Set the traffic light to the chosen phase."""
        green_phases = [0, 2]  # 0=GreenNS, 2=GreenEW
        phase = green_phases[action % len(green_phases)]

        current_phase = traci.trafficlight.getPhase(self.tl_id)

        # If switching, insert yellow phase first
        if phase != current_phase and current_phase in green_phases:
            yellow_phase = current_phase + 1
            traci.trafficlight.setPhase(self.tl_id, yellow_phase)
            for _ in range(3):
                if self._step_count < self.sim_steps:
                    traci.simulationStep()
                    self._step_count += 1

        traci.trafficlight.setPhase(self.tl_id, phase)

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        self._start_sumo()
        self._step_count = 0

        # Warm up
        for _ in range(10):
            traci.simulationStep()
            self._step_count += 1

        state = self._get_state()
        return state, {}

    def step(self, action):
        """Execute action and advance simulation."""
        self._apply_action(action)

        for _ in range(self.delta_time):
            if self._step_count >= self.sim_steps:
                break
            traci.simulationStep()
            self._step_count += 1

        state = self._get_state()
        reward = self._get_reward()

        terminated = self._step_count >= self.sim_steps
        truncated = False

        info = {
            "step": self._step_count,
            "total_waiting": sum(
                traci.lane.getWaitingTime(l)
                for l in self._incoming_lanes
            ),
        }

        return state, reward, terminated, truncated, info

    def close(self):
        """Clean up SUMO connection."""
        if self._sumo_running:
            try:
                traci.close()
            except:
                pass
            self._sumo_running = False


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing SumoTrafficEnv...")
    env = SumoTrafficEnv(sim_steps=200, delta_time=5)

    print(f"  Incoming lanes: {len(env._incoming_lanes)}")
    print(f"  Incoming edges: {len(env._incoming_edges)}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    state, info = env.reset()
    print(f"  State shape: {state.shape}")

    total_reward = 0
    steps = 0
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    env.close()
    print(f"  Episode done in {steps} steps, total reward: {total_reward:.1f}")
    print("✓ Environment test passed!")
