"""F1 Prediction Dashboard — Streamlit application.

Run from the repository root:
    streamlit run dashboard/app.py

The dashboard calls the local FastAPI server.  Start it separately with:
    uvicorn src.f1_predictor.api:app --host 127.0.0.1 --port 8000

Or configure API_BASE_URL in the sidebar to point at a remote deployment.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="F1 Prediction Dashboard",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")
    api_base = st.text_input(
        "API base URL",
        value="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI server.",
    )
    current_year = datetime.now().year
    year = st.number_input(
        "Season year", min_value=2018, max_value=2030, value=current_year, step=1
    )
    n_sims = st.slider(
        "Simulations (Monte Carlo)", min_value=200, max_value=5000, value=1000, step=200
    )
    sc_prob = st.slider("Safety-car probability", 0.0, 1.0, 0.30, 0.05)

# ---------------------------------------------------------------------------
# Helper — API calls
# ---------------------------------------------------------------------------


def _get(path: str) -> Optional[dict]:
    try:
        r = requests.get(f"{api_base}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at **{api_base}**.  Is the server running?")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


def _post(path: str, payload: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{api_base}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at **{api_base}**.  Is the server running?")
        return None
    except Exception as exc:
        st.error(f"API error ({r.status_code}): {r.text[:300]}")  # type: ignore[possibly-undefined]
        return None


# ---------------------------------------------------------------------------
# Helpers — charts
# ---------------------------------------------------------------------------


def _bar_predictions(df: pd.DataFrame, pos_col: str, title: str) -> go.Figure:
    """Horizontal bar chart of predicted positions (lower = better)."""
    df = df.sort_values(pos_col)
    fig = px.bar(
        df,
        x=pos_col,
        y="Driver",
        orientation="h",
        color="Team",
        title=title,
        labels={pos_col: "Predicted position", "Driver": ""},
        height=max(400, len(df) * 28),
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=True)
    return fig


def _podium_bar(sim_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: win / podium / top-10 % per driver."""
    df = sim_df.sort_values("Win_Pct", ascending=False).head(20).copy()
    fig = go.Figure()
    for col, label, colour in [
        ("Win_Pct", "Win %", "#FFD700"),
        ("Podium_Pct", "Podium %", "#C0C0C0"),
        ("Top10_Pct", "Top-10 %", "#CD7F32"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Bar(
                    name=label,
                    x=df["Driver"],
                    y=(df[col] * 100).round(1),
                    marker_color=colour,
                )
            )
    fig.update_layout(
        barmode="group",
        title="Win / Podium / Top-10 probability (%)",
        yaxis_title="Probability (%)",
        xaxis_title="",
        height=450,
    )
    return fig


def _position_heatmap(pos_matrix_data: dict, drivers: list[str]) -> go.Figure:
    """Heatmap of finishing-position distributions."""
    df = pd.DataFrame.from_dict(pos_matrix_data, orient="tight" if "index" in pos_matrix_data else "dict")
    if "data" in pos_matrix_data:
        df = pd.DataFrame(
            pos_matrix_data["data"],
            index=pos_matrix_data.get("index", drivers),
            columns=pos_matrix_data.get("columns", list(range(1, 21))),
        )
    # Sort drivers by median finishing position
    median_pos = (df * df.columns.astype(float)).sum(axis=1)
    df = df.loc[median_pos.sort_values().index]
    fig = px.imshow(
        df * 100,
        labels={"x": "Finishing position", "y": "Driver", "color": "Probability (%)"},
        title="Finishing-position distribution (% of simulations)",
        color_continuous_scale="Blues",
        aspect="auto",
        height=max(400, len(df) * 28),
    )
    return fig


# ---------------------------------------------------------------------------
# Main — health banner
# ---------------------------------------------------------------------------

st.title("🏎 F1 Prediction Dashboard")

health = _get("/health")
if health:
    st.success(f"API online — version {health.get('version', '?')}  |  {health.get('timestamp', '')}")
else:
    st.warning("API offline.  Start the server and refresh this page.")

# ---------------------------------------------------------------------------
# Race selector
# ---------------------------------------------------------------------------

sched_data = _get(f"/schedule/{year}")
race_names: list[str] = []
if sched_data and sched_data.get("schedule"):
    race_names = [r["EventName"] for r in sched_data["schedule"]]

race = st.selectbox(
    "Select race",
    options=race_names or ["(no schedule loaded)"],
    help="Races pulled from FastF1 via the API.",
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_race, tab_quali, tab_sim = st.tabs(["🏁 Race prediction", "⏱ Qualifying prediction", "🎲 Simulation"])

# ── Race prediction ──────────────────────────────────────────────────────────
with tab_race:
    mode = st.selectbox(
        "Prediction mode",
        ["auto", "pre_weekend", "pre_quali", "post_quali"],
        index=0,
        help="'auto' lets the model decide based on available data.",
    )
    if st.button("Predict race", key="btn_race", disabled=not race_names):
        with st.spinner("Running race prediction…"):
            data = _post("/predict/race", {"year": int(year), "race": race, "mode": mode})
        if data and data.get("predictions"):
            df = pd.DataFrame(data["predictions"])
            st.dataframe(df, use_container_width=True)
            pos_col = next(
                (c for c in ["Predicted_Race_Pos", "Predicted_Pos", "Position"] if c in df.columns),
                df.columns[0],
            )
            st.plotly_chart(_bar_predictions(df, pos_col, f"{year} {race} — Race prediction"), use_container_width=True)
        else:
            st.info("No predictions available.  Ensure models are trained (`python scripts/predict.py train`).")

# ── Qualifying prediction ────────────────────────────────────────────────────
with tab_quali:
    if st.button("Predict qualifying", key="btn_quali", disabled=not race_names):
        with st.spinner("Running qualifying prediction…"):
            data = _post("/predict/qualifying", {"year": int(year), "race": race})
        if data and data.get("predictions"):
            df = pd.DataFrame(data["predictions"])
            st.dataframe(df, use_container_width=True)
            pos_col = next(
                (c for c in ["Predicted_Quali_Pos", "Predicted_Pos", "Quali_Pos"] if c in df.columns),
                df.columns[0],
            )
            st.plotly_chart(
                _bar_predictions(df, pos_col, f"{year} {race} — Qualifying prediction"),
                use_container_width=True,
            )
        else:
            st.info("No qualifying predictions available.  Ensure qualifying model is trained.")

# ── Simulation ───────────────────────────────────────────────────────────────
with tab_sim:
    st.markdown(
        f"Run **{n_sims:,}** Monte Carlo simulations with a **{sc_prob:.0%}** safety-car probability."
    )
    if st.button("Run simulation", key="btn_sim", disabled=not race_names):
        with st.spinner(f"Simulating {n_sims} races…"):
            data = _post(
                "/simulate",
                {
                    "year": int(year),
                    "race": race,
                    "n_simulations": n_sims,
                    "sc_probability": sc_prob,
                },
            )
        if data and data.get("summary"):
            sim_df = pd.DataFrame(data["summary"])
            st.subheader("Summary")
            st.dataframe(sim_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(_podium_bar(sim_df), use_container_width=True)
            with col2:
                if data.get("position_matrix"):
                    drivers = sim_df["Driver"].tolist() if "Driver" in sim_df.columns else []
                    try:
                        st.plotly_chart(
                            _position_heatmap(data["position_matrix"], drivers),
                            use_container_width=True,
                        )
                    except Exception:
                        st.info("Position matrix chart unavailable.")
        else:
            st.info("Simulation returned no results.  Ensure models are trained.")
