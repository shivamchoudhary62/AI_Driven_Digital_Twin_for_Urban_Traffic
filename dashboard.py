"""
dashboard.py — AI Traffic Digital Twin Interactive Dashboard
Run: streamlit run dashboard.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import numpy as np
import math

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="AI Traffic Digital Twin", page_icon="🚦", layout="wide")

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(180deg, #f5f7fa 0%, #eef1f5 100%); }
    .stApp { background: linear-gradient(180deg, #f5f7fa 0%, #eef1f5 100%); }
    h1, h2, h3, h4 { color: #1a1a2e !important; }
    pre { background-color: #ffffff !important; border: 1px solid #e0e0e0 !important; border-radius: 8px !important; }
    code { color: #1a1a2e !important; background-color: #ffffff !important; }
    .stCodeBlock { background-color: #ffffff !important; }
    .hero-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #0077b6 0%, #7b2ff7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub { color: #555; font-size: 1.05rem; margin-top: 4px; }
    .metric-card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.08); border-radius: 16px;
        padding: 20px 24px; text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #0077b6; }
    .metric-label { font-size: 0.85rem; color: #666; margin-bottom: 4px; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; }
    .delta-good { color: #2e7d32; }
    .delta-bad { color: #c62828; }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #1a1a2e !important;
        border-left: 4px solid #7b2ff7; padding-left: 12px; margin: 24px 0 16px;
    }
    div[data-testid="stTabs"] button {
        color: #666 !important; font-weight: 500;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #0077b6 !important; border-bottom-color: #0077b6 !important;
    }
    .stSlider > div > div { color: #444; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown('<p class="hero-title">🚦 AI-Driven Digital Twin for Urban Traffic</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">STGCN Prediction • DRL Signal Control — Vadodara, India</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    b = pd.read_csv("data/baseline_clean.csv")
    o = pd.read_csv("data/optimized_clean.csv")
    return b, o

@st.cache_resource
def load_rf_model():
    model = joblib.load("model/traffic_model.pkl")
    encoder = joblib.load("model/label_encoder.pkl")
    features = joblib.load("model/feature_names.pkl")
    return model, encoder, features

try:
    baseline, optimized = load_data()
    rf_model, rf_encoder, rf_features = load_rf_model()
    data_loaded = True
except FileNotFoundError as e:
    st.error(f"Data files not found: {e}. Run `python main.py` first.")
    data_loaded = False
    st.stop()

# ── Compute KPIs ──────────────────────────────────────────
avg_wait_b = baseline["waiting_time"].mean()
avg_wait_o = optimized["waiting_time"].mean()
max_wait_b = baseline["waiting_time"].max()
max_wait_o = optimized["waiting_time"].max()
avg_spd_b = baseline["avg_speed_kmh"].mean()
avg_spd_o = optimized["avg_speed_kmh"].mean()
wait_imp = (avg_wait_b - avg_wait_o) / (avg_wait_b + 1e-9) * 100
spd_imp = (avg_spd_o - avg_spd_b) / (avg_spd_b + 1e-9) * 100
congested_pct = len(baseline[baseline["waiting_time"] > 10]) / len(baseline) * 100

PLOT_TEMPLATE = "plotly_white"
PLOT_BG = dict(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
COLORS = {"baseline": "#e53935", "optimized": "#2e7d32", "blue": "#1565c0",
           "purple": "#7b2ff7", "cyan": "#0077b6", "orange": "#ef6c00"}

# Vadodara road name mapping for display
ROAD_NAMES = {
    "racecourse_n": "Race Course ↑", "racecourse_s": "Race Course ↓",
    "rcdutt_e": "RC Dutt →", "rcdutt_w": "RC Dutt ←",
    "jetalpur_e": "Jetalpur →", "jetalpur_w": "Jetalpur ←",
    "gotri_n": "Gotri ↑", "gotri_s": "Gotri ↓",
    "newsama_e": "New Sama →", "newsama_w": "New Sama ←",
    "oldpadra_n": "Old Padra ↑", "oldpadra_s": "Old Padra ↓",
    "manjalpur_n": "Manjalpur ↑", "manjalpur_s": "Manjalpur ↓",
    "raopura_e": "Raopura →", "raopura_w": "Raopura ←",
    "dandia_e": "Dandia Baz →", "dandia_w": "Dandia Baz ←",
    "natubhai_loop": "Natubhai ○",
}

# ── KPI Cards ─────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Key Performance Metrics</p>', unsafe_allow_html=True)

occ_b = baseline["occupancy"].mean()
occ_o = optimized["occupancy"].mean()
occ_imp = (occ_b - occ_o) / (occ_b + 1e-9) * 100

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    ("Avg Wait Time", f"{avg_wait_o:.2f}s", f"↓ {wait_imp:.1f}%", wait_imp > 0),
    ("Max Wait Time", f"{max_wait_o:.0f}s", f"↓ {(max_wait_b-max_wait_o)/(max_wait_b+1e-9)*100:.1f}%", True),
    ("Avg Speed", f"{avg_spd_o:.1f} km/h", f"↑ {spd_imp:.1f}%", spd_imp > 0),
    ("Occupancy", f"{occ_o:.4f}", f"↓ {occ_imp:.1f}%", occ_imp > 0),
    ("DRL Queue ↓", "99.4%", "Natubhai Circle", True),
]
for col, (label, value, delta, good) in zip([c1,c2,c3,c4,c5], kpis):
    delta_class = "delta-good" if good else ("delta-bad" if good is False else "")
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta {delta_class}">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Simulation Comparison", "🧠 Live Congestion Predictor",
    "🔮 STGCN Results", "🤖 DRL Results", "🏗️ System Architecture"
])

# ── Tab 1: Simulation Comparison ─────────────────────────
with tab1:
    b_step = baseline.groupby("step").agg(
        avg_wait=("waiting_time", "mean"), avg_speed=("avg_speed_kmh", "mean"),
        avg_vc=("vehicle_count", "mean")).reset_index()
    o_step = optimized.groupby("step").agg(
        avg_wait=("waiting_time", "mean"), avg_speed=("avg_speed_kmh", "mean"),
        avg_vc=("vehicle_count", "mean")).reset_index()

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=b_step["step"], y=b_step["avg_wait"],
            name="Baseline", line=dict(color=COLORS["baseline"], width=2),
            fill="tozeroy", fillcolor="rgba(255,82,82,0.1)"))
        fig.add_trace(go.Scatter(x=o_step["step"], y=o_step["avg_wait"],
            name="AI Optimized", line=dict(color=COLORS["optimized"], width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.1)"))
        fig.add_hline(y=10, line_dash="dash", line_color="#666",
                      annotation_text="Congestion Threshold")
        fig.update_layout(template=PLOT_TEMPLATE, title="Waiting Time Over Simulation",
                          xaxis_title="Step", yaxis_title="Wait (s)", height=400,
                          legend=dict(orientation="h", y=1.12), **PLOT_BG)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=b_step["step"], y=b_step["avg_speed"],
            name="Baseline", line=dict(color=COLORS["baseline"], width=2),
            fill="tozeroy", fillcolor="rgba(255,82,82,0.1)"))
        fig2.add_trace(go.Scatter(x=o_step["step"], y=o_step["avg_speed"],
            name="AI Optimized", line=dict(color=COLORS["optimized"], width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.1)"))
        fig2.update_layout(template=PLOT_TEMPLATE, title="Speed Over Simulation",
                           xaxis_title="Step", yaxis_title="Speed (km/h)", height=400,
                           legend=dict(orientation="h", y=1.12), **PLOT_BG)
        st.plotly_chart(fig2, use_container_width=True)

    # Per-edge bar chart with Vadodara road names
    edge_b = baseline.groupby("edge_id").agg(avg_wait=("waiting_time","mean")).round(3).sort_values("avg_wait", ascending=False)
    edge_o = optimized.groupby("edge_id").agg(avg_wait=("waiting_time","mean")).round(3).reindex(edge_b.index)
    display_names = [ROAD_NAMES.get(e, e) for e in edge_b.index]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Baseline", x=display_names, y=edge_b["avg_wait"],
                          marker_color=COLORS["baseline"], opacity=0.85))
    fig3.add_trace(go.Bar(name="AI Optimized", x=display_names, y=edge_o["avg_wait"],
                          marker_color=COLORS["optimized"], opacity=0.85))
    fig3.update_layout(template=PLOT_TEMPLATE, title="Per-Road Waiting Time — Vadodara Roads",
                       barmode="group", height=420, xaxis_title="Road (Vadodara)",
                       yaxis_title="Avg Wait (s)", legend=dict(orientation="h", y=1.1),
                       xaxis_tickangle=-40, **PLOT_BG)
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("### 🧠 Live Congestion Prediction")
    st.markdown("Adjust traffic conditions below — the **Random Forest classifier** predicts if "
                "this road will become congested. This model supports real-time decision-making "
                "for traffic signal planning.")

    roads = {
        "Race Course Road": ("8627861908_318141150", 280.5),
        "RC Dutt Road": ("8451855738_8451855739", 291.9),
        "Jetalpur Road": ("8556786766_8659203468", 199.0),
        "Old Padra Road": ("5302825179_8485898828", 264.7),
        "Gotri Road": ("317076202_8526975514", 505.8),
        "Manjalpur Gate Rd": ("7865698072_7865698075", 178.8),
        "New Sama Road": ("8560186021_327366061", 188.4),
        "Natubhai Circle": ("320707451_320707452", 30.4),
        "Dandia Bazaar Road": ("8527120663_2345133351", 96.3),
        "Raopura Road": ("8527411421_2346917835", 93.7),
    }

    col_a, col_b = st.columns([1, 1])
    with col_a:
        road_name = st.selectbox("🛣️ Select Road", list(roads.keys()))
        edge_id, length_m = roads[road_name]
        hour = st.slider("🕐 Hour of Day", 0, 23, 9)
        minute = st.slider("⏱️ Minute", 0, 59, 0, step=5)
        dow = st.selectbox("📅 Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        dow_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(dow)

    with col_b:
        vc = st.slider("🚗 Vehicle Count", 0, 15, 4)
        spd = st.slider("⚡ Average Speed (km/h)", 3, 60, 30)
        wait = st.slider("⏳ Waiting Time (s)", 0, 60, 2)
        occ = st.slider("📊 Occupancy", 0.0, 0.10, 0.005, step=0.001, format="%.3f")
        cr = st.slider("🔴 Congestion Ratio", 0.3, 2.0, 0.9, step=0.05)

    # Build features
    try:
        free_flow = 45.0
        speed_ratio = spd / free_flow
        density = vc / length_m if length_m > 0 else 0
        try:
            edge_enc = rf_encoder.transform([edge_id])[0]
        except:
            edge_enc = 0

        fv = pd.DataFrame([{
            "hour_sin": round(math.sin(2*math.pi*hour/24), 4),
            "hour_cos": round(math.cos(2*math.pi*hour/24), 4),
            "dow_sin": round(math.sin(2*math.pi*dow_num/7), 4),
            "dow_cos": round(math.cos(2*math.pi*dow_num/7), 4),
            "is_weekend": 1 if dow_num >= 5 else 0,
            "edge_encoded": edge_enc, "length_m": length_m,
            "vehicle_count": vc, "avg_speed_kmh": spd,
            "waiting_time": wait, "occupancy": occ,
            "congestion_ratio": cr, "speed_ratio": round(speed_ratio, 4),
            "density": round(density, 6), "speed_rolling_3": spd,
            "wait_rolling_3": wait, "cr_rolling_3": cr, "speed_trend": 0.0,
        }])[rf_features]

        pred = rf_model.predict(fv)[0]
        prob = rf_model.predict_proba(fv)[0]

        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            if pred == 1:
                st.error(f"🔴 **CONGESTION PREDICTED** for {road_name}\n\n"
                         f"Confidence: **{prob[1]*100:.1f}%**\n\n"
                         f"The AI predicts this road will become congested in the next time window.")
            else:
                st.success(f"🟢 **NORMAL TRAFFIC** on {road_name}\n\n"
                           f"Confidence: **{prob[0]*100:.1f}%**\n\n"
                           f"No congestion expected in the near future.")

        with col_res2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=round(prob[1]*100, 1),
                title={"text": "Congestion Probability", "font": {"color": "#333"}},
                number={"suffix": "%", "font": {"color": "#1a1a2e"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#999"},
                    "bar": {"color": COLORS["cyan"]},
                    "bgcolor": "#f0f0f0",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(46,125,50,0.12)"},
                        {"range": [30, 60], "color": "rgba(239,108,0,0.12)"},
                        {"range": [60, 100], "color": "rgba(229,57,53,0.12)"},
                    ],
                    "threshold": {"line": {"color": "#e53935", "width": 3},
                                  "thickness": 0.8, "value": 50}
                }))
            fig_g.update_layout(template=PLOT_TEMPLATE, height=300,
                                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"color": "#333"})
            st.plotly_chart(fig_g, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ── Tab 3: STGCN Results ─────────────────────────────────
with tab3:
    st.markdown("### 🔮 STGCN — Spatio-Temporal Graph Neural Network")
    st.markdown("Predicts traffic speed, vehicle count, and occupancy at **T+1** using "
                "historical data across the road network graph.")

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown('<div class="metric-card"><div class="metric-label">Architecture</div>'
                  '<div class="metric-value" style="font-size:1.3rem">2× ST-Conv</div>'
                  '<div class="metric-delta">10 Vadodara roads · 4 features</div></div>',
                  unsafe_allow_html=True)
    col2.markdown('<div class="metric-card"><div class="metric-label">Parameters</div>'
                  '<div class="metric-value">96,236</div>'
                  '<div class="metric-delta">Trained on 80,640 rows</div></div>',
                  unsafe_allow_html=True)
    col3.markdown('<div class="metric-card"><div class="metric-label">Speed MAE</div>'
                  '<div class="metric-value">0.9 km/h</div>'
                  '<div class="metric-delta delta-good">Vehicle count MAE: 0.5</div></div>',
                  unsafe_allow_html=True)
    col4.markdown('<div class="metric-card"><div class="metric-label">Training Data</div>'
                  '<div class="metric-value" style="font-size:1.3rem">80,640</div>'
                  '<div class="metric-delta">real_traffic_data.csv</div></div>',
                  unsafe_allow_html=True)

    if os.path.exists("results/stgcn_evaluation.png"):
        st.image("results/stgcn_evaluation.png", caption="STGCN Training & Evaluation Results",
                 use_column_width=True)
    else:
        st.warning("Run `python models/train_stgcn.py` to generate STGCN results.")

    st.markdown("#### How STGCN Works")
    st.markdown("""
    ```
    Input: Last 12 time steps × 10 Vadodara roads × 4 features
           Features: speed, vehicle_count, occupancy, congestion_ratio
           Shape: (batch, 12, 10, 4)

    ┌─────────────────────────────────────────────────────────────┐
    │  ST-Conv Block 1                                            │
    │    Temporal Conv → Graph Conv (road adjacency) → Temporal   │
    │    + BatchNorm + Dropout                                    │
    ├─────────────────────────────────────────────────────────────┤
    │  ST-Conv Block 2                                            │
    │    Temporal Conv → Graph Conv (road adjacency) → Temporal   │
    │    + BatchNorm + Dropout                                    │
    ├─────────────────────────────────────────────────────────────┤
    │  Output Layer → Predicted state at T+1                      │
    └─────────────────────────────────────────────────────────────┘

    Output: Predicted (speed, vehicles, occupancy, congestion_ratio)
            for all 10 Vadodara roads — Shape: (batch, 10, 4)
    ```
    """)

# ── Tab 4: DRL Results ───────────────────────────────────
with tab4:
    st.markdown("### 🤖 Deep Reinforcement Learning — Traffic Light Control")
    st.markdown("DQN and PPO agents learn to control **Natubhai Circle** intersection. "
                "Combined with STGCN predictions, the DRL agent selects optimal green phases "
                "to minimize queue lengths and reduce congestion.")

    col1, col2, col3, col4 = st.columns(4)
    drl_metrics = [
        ("DQN Reward", "+41", "↑ 100%", True),
        ("PPO Reward", "+7", "↑ 100%", True),
        ("Queue Reduction", "99.4%", "Natubhai Circle", True),
        ("Baseline Reward", "-43,789", "fixed-time", None),
    ]
    for col, (label, val, delta, good) in zip([col1,col2,col3,col4], drl_metrics):
        dc = "delta-good" if good else ""
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                     f'<div class="metric-value" style="font-size:1.4rem">{val}</div>'
                     f'<div class="metric-delta {dc}">{delta}</div></div>',
                     unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        if os.path.exists("results/drl_learning_curve.png"):
            st.image("results/drl_learning_curve.png", caption="DRL Training — Learning Curves",
                     use_column_width=True)
    with col_r:
        if os.path.exists("results/drl_comparison.png"):
            st.image("results/drl_comparison.png", caption="DRL vs Fixed-Time Baseline",
                     use_column_width=True)

    if not os.path.exists("results/drl_learning_curve.png"):
        st.warning("Run `python models/train_drl.py` to generate DRL results.")

    st.markdown("#### MDP Formulation — Natubhai Circle")
    mdp_col1, mdp_col2, mdp_col3 = st.columns(3)
    with mdp_col1:
        st.markdown("""
        **State (13-dim)**
        - Queue lengths on 6 incoming lanes
        - Current traffic light phase
        - Density on 3 edges
        - Speed on 3 edges
        """)
    with mdp_col2:
        st.markdown("""
        **Action (2)**
        - 0: Green North-South
        - 1: Green East-West
        - Auto yellow transitions (3 steps)
        """)
    with mdp_col3:
        st.markdown(r"""
        **Reward**

        $R_t = -\sum_{j=1}^{N} q_{j,t} + 0.1 \times \text{throughput}$

        Minimize queues, maximize flow.
        """)

# ── Tab 5: Architecture ──────────────────────────────────
with tab5:
    st.markdown("### 🏗️ System Architecture")

    st.markdown("#### Vadodara Road Network")

    # ── Interactive Plotly Node-Edge Graph ──
    # Junction coordinates (scaled for visualization)
    nodes = {
        "Natubhai Circle":   {"x": 300, "y": 300, "type": "traffic_light"},
        "Prodmore":          {"x": 550, "y": 500, "type": "traffic_light"},
        "Karelibaug":        {"x": 550, "y": 300, "type": "traffic_light"},
        "Sama Junction":     {"x": 800, "y": 300, "type": "traffic_light"},
        "Raopura Junction":  {"x": 100, "y": 300, "type": "traffic_light"},
        "Manjalpur":         {"x": 100, "y": 100, "type": "traffic_light"},
        "Gotri End":         {"x": 800, "y": 550, "type": "terminal"},
        "New Sama End":      {"x": 1000, "y": 300, "type": "terminal"},
        "Old Padra End":     {"x": 100, "y": -50, "type": "terminal"},
    }
    edges = [
        ("Natubhai Circle", "Prodmore",         "Race Course Rd",   "280.5m"),
        ("Prodmore",        "Karelibaug",        "RC Dutt Rd",       "291.9m"),
        ("Natubhai Circle", "Karelibaug",        "Dandia Bazaar Rd", "96.3m"),
        ("Karelibaug",      "Sama Junction",     "Jetalpur Rd",      "199.0m"),
        ("Sama Junction",   "Gotri End",         "Gotri Rd",         "505.8m"),
        ("Sama Junction",   "New Sama End",      "New Sama Rd",      "188.4m"),
        ("Natubhai Circle", "Raopura Junction",  "Raopura Rd",       "93.7m"),
        ("Raopura Junction","Manjalpur",          "Manjalpur Gate Rd","178.8m"),
        ("Manjalpur",       "Old Padra End",     "Old Padra Rd",     "264.7m"),
        ("Natubhai Circle", "Raopura Junction",  "Natubhai Loop",    "30.4m"),
    ]

    fig_net = go.Figure()

    # Draw edges (roads)
    edge_colors = ["#e53935","#1565c0","#ef6c00","#2e7d32","#7b2ff7",
                   "#0077b6","#d81b60","#00897b","#5e35b1","#bdbdbd"]
    for i, (n1, n2, road_name, dist) in enumerate(edges):
        x0, y0 = nodes[n1]["x"], nodes[n1]["y"]
        x1, y1 = nodes[n2]["x"], nodes[n2]["y"]
        color = edge_colors[i % len(edge_colors)]

        # Curve the Natubhai Loop so it doesn't overlap with Raopura Rd
        if road_name == "Natubhai Loop":
            # Draw as arc below the straight Raopura Rd line
            import numpy as np
            t_vals = np.linspace(0, 1, 30)
            cx, cy = (x0+x1)/2, (y0+y1)/2 - 60  # control point below
            bx = (1-t_vals)**2*x0 + 2*(1-t_vals)*t_vals*cx + t_vals**2*x1
            by = (1-t_vals)**2*y0 + 2*(1-t_vals)*t_vals*cy + t_vals**2*y1
            mx, my = cx, cy + 20  # label position
            fig_net.add_trace(go.Scatter(
                x=bx.tolist(), y=by.tolist(), mode="lines",
                line=dict(color=color, width=2.5, dash="dot"),
                hoverinfo="text", hovertext=f"<b>{road_name}</b><br>Length: {dist}<br>{n1} ↔ {n2}",
                showlegend=False,
            ))
        else:
            mx, my = (x0+x1)/2, (y0+y1)/2
            fig_net.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="text", hovertext=f"<b>{road_name}</b><br>Length: {dist}<br>{n1} ↔ {n2}",
                showlegend=False,
            ))

        # Edge label
        fig_net.add_annotation(
            x=mx, y=my, text=f"<b>{road_name}</b><br>{dist}",
            showarrow=False, font=dict(size=9, color=color, family="Inter"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1,
            borderpad=3,
        )

    # Draw nodes (junctions)
    for name, info in nodes.items():
        is_tl = info["type"] == "traffic_light"
        fig_net.add_trace(go.Scatter(
            x=[info["x"]], y=[info["y"]], mode="markers+text",
            marker=dict(
                size=28 if is_tl else 20,
                color="#0077b6" if is_tl else "#7b2ff7",
                symbol="circle",
                line=dict(width=2.5, color="#ffffff"),
            ),
            text=name, textposition="top center",
            textfont=dict(size=11, color="#1a1a2e", family="Inter"),
            hoverinfo="text",
            hovertext=f"<b>{name}</b><br>Type: {'🚦 Traffic Light' if is_tl else '📍 Terminal'}<br>"
                      f"Pos: ({info['x']}, {info['y']})",
            showlegend=False,
        ))

    # Legend-like markers
    fig_net.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="🚦 Traffic Light Junction",
        marker=dict(size=14, color="#0077b6", symbol="circle", line=dict(width=2, color="#fff")),
    ))
    fig_net.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="📍 Terminal Node",
        marker=dict(size=12, color="#7b2ff7", symbol="circle", line=dict(width=2, color="#fff")),
    ))

    fig_net.update_layout(
        template="plotly_white", height=520,
        title=dict(text="Vadodara Road Network — 9 Junctions · 10 Roads",
                   font=dict(size=15, color="#1a1a2e", family="Inter")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1, fixedrange=True),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center",
                    font=dict(size=11)),
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="rgba(245,247,250,0.5)",
    )
    st.plotly_chart(fig_net, use_container_width=True)

    # Save network graph to results
    try:
        fig_net.write_image("results/vadodara_network.png", width=1200, height=600, scale=2)
    except Exception:
        pass  # kaleido not available

    st.markdown("**6 traffic-light intersections** · **19 bidirectional edges** · **10 Vadodara roads** · **~1,298 vehicles/hour**")

    st.markdown("#### Integrated AI Pipeline")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │              INTEGRATED AI TRAFFIC OPTIMIZER                 │
    ├──────────────────────────────┬───────────────────────────────┤
    │  STGCN (Forecaster)          │  DRL — PPO/DQN (Controller)   │
    │                              │                               │
    │  Predicts speed, density,    │  Controls traffic light       │
    │  occupancy at T+1 across     │  phases at Natubhai Circle    │
    │  all 10 Vadodara roads       │  using real-time observations │
    ├──────────────────────────────┴───────────────────────────────┤
    │  Flow: Observe traffic → STGCN predicts future state →       │
    │        DRL selects optimal green phase → Reduce delays       │
    └──────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("#### Pipeline Steps")
    steps = [
        ("1️⃣", "Baseline", "SUMO fixed-time simulation"),
        ("2️⃣", "STGCN", "Predict future traffic state"),
        ("3️⃣", "DRL Agent", "Learn optimal signal timing"),
        ("4️⃣", "Integrate", "Predict + control in one sim"),
        ("5️⃣", "Evaluate", "Compare vs baseline"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f'<div class="metric-card" style="min-height:140px">'
                     f'<div style="font-size:1.8rem">{icon}</div>'
                     f'<div class="metric-label" style="font-size:1rem;color:#1a1a2e;'
                     f'font-weight:600;margin:6px 0">{title}</div>'
                     f'<div style="font-size:0.8rem;color:#555">{desc}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("#### Tech Stack")
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.info("**Simulation**\n\nSUMO 1.26.0\nTraCI API\n10 Vadodara Roads\n6 Traffic Lights")
    tc2.info("**ML/DL**\n\nPyTorch (STGCN)\nScikit-learn (RF)\nStable-Baselines3 (DRL)")
    tc3.info("**Frontend**\n\nStreamlit\nPlotly\nCustom Light Theme")
    tc4.info("**Data**\n\n80,640 rows\n10 roads\n4 features\n7-day coverage")

st.markdown("---")
st.caption("AI-Driven Digital Twin for Urban Traffic Optimization • "
           "STGCN + DRL • Built with SUMO, PyTorch, Streamlit")