from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure `src/` is importable when running `streamlit run ...` from repo root.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import fastf1

from f1_predictor.config import config
from f1_predictor.prediction import F1Predictor


def _enable_fastf1_cache() -> None:
    """Enable FastF1 cache using the project config."""
    try:
        fastf1.Cache.enable_cache(config.get("paths.cache_dir"))
    except Exception:
        # Fallback: FastF1 will still work but will re-download data more often.
        pass


@st.cache_data(show_spinner=False)
def _load_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(int(year), include_testing=False).copy()
    schedule.sort_values("RoundNumber", inplace=True)
    return schedule


def _setup_checklist() -> None:
    st.subheader("Setup checklist")
    races_csv = config.get("paths.races_csv")
    quali_csv = config.get("paths.quali_csv")
    models_dir = config.get("paths.models_dir")

    checks = [
        ("Historical race CSV exists", Path(races_csv).exists(), races_csv),
        ("Historical quali CSV exists", Path(quali_csv).exists(), quali_csv),
        ("Models directory exists", Path(models_dir).exists(), models_dir),
        ("Qualifying model exists", Path(models_dir, "qualifying_model.pkl").exists(), str(Path(models_dir, "qualifying_model.pkl"))),
        ("Race model exists", Path(models_dir, "race_model.pkl").exists(), str(Path(models_dir, "race_model.pkl"))),
    ]
    for label, ok, detail in checks:
        st.write(f"- {'✅' if ok else '❌'} **{label}** — `{detail}`")

    if not all(ok for _, ok, _ in checks[:2]):
        st.info("Run: `python scripts/predict.py fetch-data --force`")
    if not all(ok for _, ok, _ in checks[3:]):
        st.info("Run: `python scripts/predict.py train`")


def _render_predictions_tab() -> None:
    st.header("Predictions")
    _setup_checklist()

    st.divider()
    st.subheader("Choose an event")

    default_year = int(datetime.now().year)
    year = st.number_input("Season year", min_value=2010, max_value=2100, value=default_year, step=1)

    try:
        schedule = _load_schedule(int(year))
        event_names = schedule["EventName"].astype(str).tolist()
    except Exception as e:
        st.error(f"Failed to load schedule for {year}: {e}")
        return

    race_name = st.selectbox("Race (EventName)", options=event_names, index=0)
    session = st.radio("Session", options=["qualifying", "race"], horizontal=True, index=1)
    mode = "auto"
    if session == "race":
        mode = st.selectbox(
            "Race mode (how to use qualifying)",
            options=["auto", "pre-weekend", "pre-quali", "post-quali"],
            index=0,
        )

    st.divider()
    run = st.button("Run prediction", type="primary")

    if not run:
        return

    _enable_fastf1_cache()
    predictor = F1Predictor()

    with st.spinner("Running prediction..."):
        if session == "qualifying":
            res = predictor.predict_qualifying(int(year), str(race_name), scenario="qualifying")
        else:
            res = predictor.predict_race(int(year), str(race_name), mode=str(mode).replace("-", "_"))

    if res is None or (hasattr(res, "empty") and res.empty):
        st.error("No prediction produced. Check logs and ensure data/models exist.")
        return

    st.success("Prediction complete.")
    st.dataframe(res, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name=f"{session}_predictions_{year}_{race_name.replace(' ', '_')}.csv",
        mime="text/csv",
    )


def _telemetry_plot(title: str, series_by_driver: dict[str, pd.DataFrame]) -> None:
    fig = go.Figure()
    for drv, df in series_by_driver.items():
        fig.add_trace(
            go.Scatter(
                x=df["Distance"],
                y=df["Speed"],
                mode="lines",
                name=drv,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        legend_title="Driver",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_telemetry_tab() -> None:
    st.header("Telemetry explorer (FastF1)")
    st.caption("Fetches telemetry via FastF1 and uses the local FastF1 cache for speed.")

    default_year = int(datetime.now().year)
    year = st.number_input("Telemetry season year", min_value=2010, max_value=2100, value=default_year, step=1)

    try:
        schedule = _load_schedule(int(year))
        event_names = schedule["EventName"].astype(str).tolist()
    except Exception as e:
        st.error(f"Failed to load schedule for {year}: {e}")
        return

    race_name = st.selectbox("Telemetry race (EventName)", options=event_names, index=0, key="telemetry_race")
    session_code = st.selectbox(
        "Session",
        options=["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"],
        index=6,
    )

    load = st.button("Load session & telemetry", type="primary", key="load_session")
    if not load:
        st.info("Pick an event/session and click **Load session & telemetry**.")
        return

    _enable_fastf1_cache()

    with st.spinner("Loading session (this can take a bit on first run)..."):
        ses = fastf1.get_session(int(year), str(race_name), str(session_code))
        ses.load(laps=True, telemetry=True, weather=False, messages=False)

    if ses.results is None or ses.results.empty:
        st.error("No results found for this session.")
        return

    drivers = sorted(ses.results["Abbreviation"].astype(str).tolist())
    c1, c2 = st.columns(2)
    with c1:
        d1 = st.selectbox("Driver 1", options=drivers, index=0)
    with c2:
        d2 = st.selectbox("Driver 2 (optional)", options=["(none)"] + drivers, index=0)

    def _get_fastest_speed_trace(driver: str) -> pd.DataFrame | None:
        try:
            laps = ses.laps.pick_driver(driver)
            if laps is None or laps.empty:
                return None
            lap = laps.pick_fastest()
            car = lap.get_car_data()
            car = car.add_distance()
            out = car[["Distance", "Speed"]].copy()
            out["Distance"] = pd.to_numeric(out["Distance"], errors="coerce")
            out["Speed"] = pd.to_numeric(out["Speed"], errors="coerce")
            out.dropna(subset=["Distance", "Speed"], inplace=True)
            return out
        except Exception:
            return None

    with st.spinner("Extracting fastest-lap telemetry..."):
        series = {}
        t1 = _get_fastest_speed_trace(str(d1))
        if t1 is not None and not t1.empty:
            series[str(d1)] = t1
        if d2 != "(none)":
            t2 = _get_fastest_speed_trace(str(d2))
            if t2 is not None and not t2.empty:
                series[str(d2)] = t2

    if not series:
        st.error("Could not extract telemetry for the selected driver(s).")
        return

    _telemetry_plot(
        title=f"{year} {race_name} {session_code} — fastest lap speed trace",
        series_by_driver=series,
    )


def main() -> None:
    st.set_page_config(page_title="F1 Predictor", layout="wide")
    st.title("F1 Predictor — Qualifying & Race")

    tabs = st.tabs(["Predictions", "Telemetry"])
    with tabs[0]:
        _render_predictions_tab()
    with tabs[1]:
        _render_telemetry_tab()


if __name__ == "__main__":
    main()

