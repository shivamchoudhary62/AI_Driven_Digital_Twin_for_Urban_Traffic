# Presentation Prompt — 15 Slides (Updated)

> Copy-paste the prompt below into any AI presentation tool (Gamma, SlidesAI, Beautiful.ai, ChatGPT, etc.) to generate your slides.

---

## Prompt

```
Create a professional, modern academic presentation of 15 slides for the following project. Use a clean dark/navy theme with teal and purple accents. Include relevant icons and diagrams where described.

---

SLIDE 1 — TITLE SLIDE
Title: "AI-Driven Digital Twin for Urban Traffic Optimization"
Subtitle: "STGCN Prediction + Deep Reinforcement Learning Signal Control"
Course: Semester 6 — AI-Driven Smart Traffic & Logistics
Team Members:
  • Ashish Kumar (23AI016)
  • Ishan (23AI027)
  • Pratyush Rai (23AI039)
  • Shivam Kumar (23AI050)
Include a small traffic/city icon.

---

SLIDE 2 — PROBLEM STATEMENT
Title: "The Urban Traffic Challenge"
Content:
  • Indian cities face severe traffic congestion — Vadodara, Gujarat has 10+ major roads converging at key intersections
  • Fixed-time traffic signals cannot adapt to real-time traffic conditions
  • Result: long waiting times, low average speed, high occupancy, wasted fuel
  • Need: An AI system that can PREDICT congestion and OPTIMIZE signal timing — trained on synthetic data now, designed for real-time API data in production
Add a visual showing congested traffic vs free-flowing traffic.

---

SLIDE 3 — WHAT IS A DIGITAL TWIN?
Title: "Digital Twin Concept"
Content:
  • A digital twin is a virtual replica of a real-world system
  • Our digital twin replicates Vadodara's road network in SUMO (Simulation of Urban Mobility)
  • Currently trained on synthetic data generated from real Vadodara road profiles
  • Architecture is designed to ingest real-time API data (Google Maps, TomTom, IoT sensors) for continuous model retraining
  • AI models run inside the twin to predict and optimize — then results transfer to the real world
Show a diagram: Real World (Vadodara) ←→ Digital Twin (SUMO Simulator) ←→ AI Models ←→ [Future: Real-Time API Feeds]

---

SLIDE 4 — SYSTEM ARCHITECTURE (Overview)
Title: "System Architecture"
Content — show a flow diagram:
  1. SUMO Simulator (Vadodara Network) produces Baseline Data (fixed-time signals)
  2. Synthetic Training Data (80,640 rows, generated from real Vadodara road profiles) feeds into → Random Forest (Congestion Classifier) + STGCN (Spatio-Temporal GNN)
  3. STGCN predicts future traffic state at T+1
  4. DRL Agent (PPO/DQN) controls traffic signals using predictions
  5. Integrated Optimizer runs both models in one SUMO simulation → Optimized Data
  6. Results compared: Baseline vs AI-Optimized
  7. Everything visualized in a Streamlit Dashboard
  Note: Architecture is designed to swap synthetic data with real-time API feeds (Google Maps / TomTom / municipal sensors) for continuous model retraining in production.

---

SLIDE 5 — VADODARA ROAD NETWORK
Title: "Vadodara Road Network — SUMO Simulation"
Content:
  • 10 real Vadodara roads modeled: Race Course Road, RC Dutt Road, Jetalpur Road, Gotri Road, New Sama Road, Dandia Bazaar Road, Natubhai Circle, Raopura Road, Manjalpur Gate Road, Old Padra Road
  • 9 junctions (6 with traffic lights)
  • 19 bidirectional edges
  • ~1,298 vehicles over a 1-hour simulation (3,600 time steps)
  • Road lengths range from 30.4m (Natubhai roundabout) to 505.8m (Gotri Road)
Show a node-edge graph diagram of the network with junction names and road distances.

---

SLIDE 6 — DATA PIPELINE
Title: "Data Pipeline — Synthetic Now, Real-Time API Ready"
Content:
  • Currently uses synthetic data generated from real Vadodara road profiles (seed: 360 real observations)
  • Extracted per-road profiles (speed mean/std, vehicle count, congestion ratio, waiting time, occupancy)
  • Generated 80,640 synthetic training rows spanning 4 weeks × 7 days × 288 time slots × 10 roads
  • Realistic Indian city traffic curves: weekday peaks (8-10 AM, 5-8 PM), weekend shifts, night lulls
  • Target label: is_congested_next (binary) — congested if congestion_ratio > 1.1 OR speed < 15 km/h OR waiting_time > 10s
  • Dataset split: 80% train, 20% test
  • FUTURE READY: Data pipeline designed to swap synthetic generation with real-time traffic API data (Google Maps Traffic API / TomTom / municipal IoT sensors) for continuous model retraining and live adaptation

---

SLIDE 7 — MODEL 1: STGCN (Spatio-Temporal Graph Convolutional Network)
Title: "STGCN — Traffic State Prediction"
Content:
  • Purpose: Predict future traffic state (speed, vehicle count, occupancy, congestion ratio) at T+1 for all 10 roads
  • Architecture: 2× ST-Conv Blocks (Temporal Conv → Graph Conv → Temporal Conv) + Output FC Layer
  • Input: Last 12 time steps × 10 roads × 4 features = (batch, 12, 10, 4)
  • Output: Predicted state at T+1 = (batch, 10, 4)
  • Graph Conv uses normalized adjacency matrix built from real Vadodara road connectivity
  • Parameters: 96,236
  • Training: 80 epochs, Adam optimizer (lr=0.001), Cosine Annealing scheduler, MSE loss
Show a block diagram: Input → [Temporal Conv → Graph Conv → Temporal Conv] ×2 → Output Layer → Prediction

---

SLIDE 8 — STGCN RESULTS
Title: "STGCN Prediction Accuracy"
Content:
  • Speed MAE: 0.927 km/h (very accurate!)
  • Vehicle Count MAE: 0.532
  • Occupancy MAE: 0.003
  • Congestion Ratio MAE: 0.074
  • Best Test MSE: 0.2749
  • Training converges in ~10 epochs, best model selected by test loss
Show 4 evaluation plots: (1) Training/Test Loss curve, (2) Predicted vs Actual speed for Jetalpur Road, (3) Scatter plot of predicted vs actual speed for all roads, (4) Per-road MAE bar chart.

---

SLIDE 9 — MODEL 2: DRL (Deep Reinforcement Learning)
Title: "DRL — Traffic Light Control"
Content — MDP Formulation for Natubhai Circle:
  • State (13-dimensional): Queue lengths on 6 incoming lanes + Current traffic light phase + Density on 3 edges + Average speed on 3 edges
  • Action Space: {0: Green North-South, 1: Green East-West} with automatic 3-step yellow transitions
  • Reward: R_t = -Σ(queue_lengths) + 0.1 × throughput — minimize queues, maximize flow
  • Algorithms trained: DQN (lr=1e-3, replay buffer=50K, exploration=30%) AND PPO (lr=3e-4, 128 steps, 10 epochs, clip=0.2)
  • Environment: Custom Gymnasium env wrapping SUMO via TraCI API
Show MDP diagram: State → Agent → Action → Environment → Reward → State

---

SLIDE 10 — DRL RESULTS
Title: "DRL Agent Performance"
Content:
  • DQN Final Reward: +41 (vs baseline -43,789)
  • PPO Final Reward: +7 (vs baseline -43,789)
  • Queue Reduction at Natubhai Circle: 99.4%
  • Both agents massively outperform fixed-time baseline
  • PPO agent selected for integrated optimizer (more stable)
Show: (1) Learning curves (DQN and PPO reward over episodes), (2) Bar chart comparing baseline vs DQN vs PPO total rewards, (3) Queue length over time comparison.

---

SLIDE 11 — MODEL 3: RANDOM FOREST (Congestion Classifier)
Title: "Random Forest — Live Congestion Predictor"
Content:
  • Purpose: Predict whether a road will become congested in the NEXT time window (binary classification)
  • 18 Features: Time-cyclic (hour/dow sin/cos), is_weekend, edge_encoded, length, vehicle_count, avg_speed, waiting_time, occupancy, congestion_ratio, speed_ratio, density, 3 rolling averages, speed_trend
  • Model: 200 trees, max_depth=15, balanced class weights
  • Used in the Streamlit dashboard for real-time interactive predictions
Show: Feature importance bar chart + Confusion matrix.

---

SLIDE 12 — INTEGRATED OPTIMIZER
Title: "Integrated AI Traffic Optimizer"
Content — how STGCN + DRL work together in one simulation:
  • STGCN predicts future traffic state every 10 simulation steps (fills 12-step sliding window buffer)
  • DRL agent controls Natubhai Circle traffic lights every 5 steps using real-time observations
  • Pipeline: Observe traffic → STGCN predicts congestion at T+1 → DRL selects optimal green phase → Reduce delays
  • Single SUMO simulation run (3,600 steps, 1 hour)
  • Result: 720 DRL signal actions, 349 STGCN predictions per run
  • Output: data/optimized_clean.csv for comparison against baseline
Show flow: SUMO → [Every 10 steps: STGCN Predict] → [Every 5 steps: DRL Act] → Record → Compare

---

SLIDE 13 — FINAL RESULTS
Title: "AI Optimization Results — Baseline vs AI-Optimized"
Content — show comparison table:
  | Metric           | Baseline  | AI-Optimized | Improvement |
  |------------------|-----------|--------------|-------------|
  | Avg Wait Time    | 6.48s     | 3.90s        | ↓ 39.8%     |
  | Max Wait Time    | 352s      | 240s         | ↓ 31.8%     |
  | Avg Speed        | 37.89 km/h| 39.92 km/h   | ↑ 5.4%      |
  | Avg Occupancy    | 0.0266    | 0.0200       | ↓ 24.8%     |
  | Queue (Natubhai) | High      | Near-zero    | ↓ 99.4%     |
Show: (1) Waiting time over simulation (baseline red vs optimized green), (2) Speed over simulation, (3) Per-road waiting time bar chart.

---

SLIDE 14 — INTERACTIVE DASHBOARD
Title: "Streamlit Dashboard — Live Demo"
Content:
  • 5-tab interactive dashboard built with Streamlit + Plotly
  • Tab 1: Simulation Comparison — time-series plots + per-road bars
  • Tab 2: Live Congestion Predictor — adjust sliders, RF predicts congestion in real-time with gauge
  • Tab 3: STGCN Results — architecture details + evaluation plots
  • Tab 4: DRL Results — MDP formulation, learning curves, reward metrics
  • Tab 5: System Architecture — interactive Plotly network graph, pipeline cards, tech stack
Show screenshots of the dashboard (KPI cards, gauge chart, comparison plots).

---

SLIDE 15 — CONCLUSION & FUTURE VISION
Title: "Conclusion & Future Vision — Real-Time API Integration"
Content:
  Key Achievements:
  • Built an end-to-end AI-powered Digital Twin for Vadodara traffic
  • STGCN predicts traffic with <1 km/h speed error
  • DRL reduces queue lengths by 99.4% at controlled intersections
  • Overall 39.8% reduction in average waiting time
  • Interactive dashboard for real-time monitoring and prediction

  Tech Stack:
  • Simulation: SUMO 1.26.0 + TraCI API
  • Deep Learning: PyTorch (STGCN — 96K params)
  • Reinforcement Learning: Stable-Baselines3 (DQN + PPO)
  • Machine Learning: Scikit-learn (Random Forest — 200 trees)
  • Dashboard: Streamlit + Plotly
  • Data: Currently synthetic (80,640 rows) — architecture ready for real-time API data

  Future Work — Real-Time API Integration:
  • Connect to real-time traffic APIs (Google Maps Traffic API / TomTom / municipal IoT sensors) to replace synthetic data
  • Enable continuous model retraining — STGCN and RF auto-update as new real data flows in
  • Live dashboard fed by streaming API data for city-wide real-time traffic monitoring
  • Extend DRL to multiple intersections (multi-agent reinforcement learning)
  • Add vehicle rerouting recommendations using STGCN predictions

---

Style guidelines: Use a professional dark navy/charcoal background with white text. Use teal (#0077b6) and purple (#7b2ff7) as accent colors. Keep bullet points concise. Use icons for each section. Make diagrams clean and clear. Font: Inter or similar modern sans-serif.
```

---

> [!TIP]
> You can paste this prompt into **Gamma.app** (free, generates beautiful slides from prompts), **SlidesAI**, **Beautiful.ai**, or **ChatGPT with canvas** to auto-generate the presentation.

> [!NOTE]
> The team member names are from your earlier ASP project. Update them if the team is different for this project.
