"""
=============================================================================
 STEP 2 of 2 — THE DASHBOARD
=============================================================================

 What this does:
   - Asks the free Open-Meteo weather service for the next 1-3 days of weather
   - Feeds that weather into the model you trained in Step 1
   - Shows the predicted power, and tells the grid operator what to do about it

 How to run it:
       streamlit run app.py

 A browser tab opens automatically. Leave this terminal window running.

 IF THE INTERNET FAILS during your demo, the app switches to a built-in
 offline weather pattern and keeps working. It will say so on screen.
 Your demo cannot die because of wifi.
=============================================================================
"""

import os
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
import plotly.graph_objects as go

IST = timezone(timedelta(hours=5, minutes=30))
OUT_DIR = "outputs"

st.set_page_config(page_title="Solar Generation Forecasting",
                   layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Load the trained model (cached so it loads once, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(OUT_DIR, "solar_model.json")
    info_path = os.path.join(OUT_DIR, "model_info.json")
    if not os.path.exists(model_path):
        return None, None
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(info_path) as f:
        info = json.load(f)
    return booster, info


booster, info = load_model()
if booster is None:
    st.error("No trained model found. Run this first:   python train_model.py")
    st.stop()


# ---------------------------------------------------------------------------
# Sun position -- same pure-astronomy maths as the training script
# ---------------------------------------------------------------------------
def solar_elevation(ts_index, lat, lon, tz_offset_hours=5.5):
    ts = pd.DatetimeIndex(ts_index)
    doy = ts.dayofyear.values
    hour = ts.hour.values + ts.minute.values / 60.0
    decl = np.radians(23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.0)))
    b = np.radians(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_time = hour + (4.0 * (lon - 15.0 * tz_offset_hours) + eot) / 60.0
    ha = np.radians(15.0 * (solar_time - 12.0))
    lat_r = np.radians(lat)
    sin_elev = (np.sin(lat_r) * np.sin(decl)
                + np.cos(lat_r) * np.cos(decl) * np.cos(ha))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))


# ---------------------------------------------------------------------------
# Weather: try the internet, fall back to a clear-sky pattern if it fails.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900)
def fetch_weather(lat, lon, hours):
    """Returns (dataframe, source_label). Never raises."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "shortwave_radiation,temperature_2m,wind_speed_10m,cloud_cover",
                "forecast_days": max(2, int(np.ceil(hours / 24)) + 1),
                "timezone": "Asia/Kolkata",
            },
            timeout=12,
        )
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({
            "ts": pd.to_datetime(h["time"]),
            "ghi": h["shortwave_radiation"],
            "temp": h["temperature_2m"],
            "wind": h["wind_speed_10m"],
            "cloud": h["cloud_cover"],
        }).dropna()
        return df, "Live forecast (Open-Meteo)"
    except Exception:
        # Offline fallback so the demo never dies.
        start = pd.Timestamp.now(tz=IST).floor("h").tz_localize(None)
        ts = pd.date_range(start, periods=hours + 24, freq="h")
        elev = solar_elevation(ts, lat, lon)
        ghi = np.clip(np.sin(np.radians(np.clip(elev, 0, None))) * 1000, 0, None)
        return pd.DataFrame({
            "ts": ts, "ghi": ghi,
            "temp": 26 + 9 * ghi / 1000, "wind": 2.5,
            "cloud": 20.0,
        }), "OFFLINE fallback (clear-sky estimate)"


def build_forecast(weather, lat, lon, hours):
    """Turn hourly weather into a 15-minute power forecast."""
    w = weather.set_index("ts").resample("15min").interpolate("time")
    w = w.iloc[: hours * 4]
    ts = w.index

    elev = solar_elevation(ts, lat, lon)
    ghi = np.clip(w["ghi"].values, 0, None)
    ghi[elev <= 0] = 0.0
    ambient = w["temp"].values
    wind = np.clip(w["wind"].values, 0.1, None)

    # Estimate panel temperature from air temperature, sunshine and wind.
    # Panels run hotter than the air, and hot panels are less efficient.
    module = ambient + ghi / (25.0 + 6.84 * wind)

    slot = ts.hour * 4 + ts.minute // 15
    feats = pd.DataFrame({
        "irradiance_wm2": ghi,
        "ambient_temp_c": ambient,
        "module_temp_c": module,
        "irr_roll_1h": pd.Series(ghi).rolling(4, min_periods=1).mean().values,
        "irr_roll_3h": pd.Series(ghi).rolling(12, min_periods=1).mean().values,
        "temp_diff": module - ambient,
        "irr_x_temp": ghi * ambient / 1000.0,
        "solar_elevation": elev,
        "tod_sin": np.sin(2 * np.pi * slot / 96),
        "tod_cos": np.cos(2 * np.pi * slot / 96),
        "doy_sin": np.sin(2 * np.pi * ts.dayofyear / 365),
        "doy_cos": np.cos(2 * np.pi * ts.dayofyear / 365),
    })[info["features"]]

    pred = booster.predict(xgb.DMatrix(feats))
    pred = np.clip(pred, 0, info["rated_capacity_kw"])
    pred[elev <= 0] = 0.0          # never predict power at night

    band = 0.12 * pred + np.where(elev > 0, 0.02 * info["rated_capacity_kw"], 0)
    return pd.DataFrame({
        "ts": ts, "pred_mw": pred / 1000.0,
        "lower_mw": np.clip(pred - band, 0, None) / 1000.0,
        "upper_mw": (pred + band) / 1000.0,
        "daylight": elev > 0,
    })


# ---------------------------------------------------------------------------
# Decision engine -- plain rules, no machine learning
# ---------------------------------------------------------------------------
def make_recommendations(fc, baseline_mw, battery_mw):
    recs = []
    p = fc["pred_mw"].values
    ts = fc["ts"].values

    surplus = p - baseline_mw
    over = surplus > battery_mw
    if over.any():
        i = np.argmax(np.where(over, surplus, -np.inf))
        recs.append(("CRITICAL", pd.Timestamp(ts[i]),
                     f"Output exceeds demand plus battery headroom by "
                     f"{surplus[i] - battery_mw:.1f} MW",
                     f"CURTAIL approximately {100 * (surplus[i] - battery_mw) / max(p[i], 0.1):.0f}% of output"))

    charge = (surplus > 0) & ~over
    if charge.any():
        idx = np.where(charge)[0]
        recs.append(("INFO", pd.Timestamp(ts[idx[0]]),
                     f"Surplus generation from {pd.Timestamp(ts[idx[0]]):%H:%M} "
                     f"to {pd.Timestamp(ts[idx[-1]]):%H:%M}",
                     f"CHARGE battery storage, up to {min(surplus[idx].max(), battery_mw):.1f} MW"))

    # Steep drops, measured over a one-hour window
    if len(p) > 4:
        ramp = pd.Series(p).diff(4).values
        j = int(np.nanargmin(ramp))
        if ramp[j] < -0.15 * max(p.max(), 0.1):
            recs.append(("WARNING", pd.Timestamp(ts[j]),
                         f"Sharp fall of {abs(ramp[j]):.1f} MW within one hour",
                         "PRE-DISCHARGE battery storage before this window"))

    deficit = (p < baseline_mw * 0.5) & fc["daylight"].values
    if deficit.any():
        i = int(np.where(deficit)[0][0])
        recs.append(("WARNING", pd.Timestamp(ts[i]),
                     f"Daytime output falls to {p[i]:.1f} MW against a "
                     f"{baseline_mw:.1f} MW baseline",
                     "PLACE peaker plant on standby"))
    return recs


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")
horizon = st.sidebar.radio("Forecast horizon", [24, 48, 72],
                           index=1, format_func=lambda h: f"{h} hours")
st.sidebar.markdown("---")
baseline_mw = st.sidebar.slider("Grid demand baseline (MW)", 0.0, 40.0, 18.0, 0.5)
battery_mw = st.sidebar.slider("Battery capacity (MW)", 0.0, 20.0, 6.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.caption("Plant location")
lat = st.sidebar.number_input("Latitude", value=float(info["latitude"]), format="%.4f")
lon = st.sidebar.number_input("Longitude", value=float(info["longitude"]), format="%.4f")

st.sidebar.markdown("---")
st.sidebar.caption("Model accuracy on unseen days")
st.sidebar.metric("WAPE (daylight)", f"{info['wape_daylight']:.2f}%")
st.sidebar.metric("Average miss", f"{info['mae_kw']:,.0f} kW")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("AI-Powered Renewable Generation Forecasting")
st.caption("Utility-scale solar - 15-minute grid blocks - Asia/Kolkata")

weather, source = fetch_weather(lat, lon, horizon)
fc = build_forecast(weather, lat, lon, horizon)

if source.startswith("OFFLINE"):
    st.warning(f"{source} - live weather unreachable, using a clear-sky estimate.")
else:
    st.success(f"{source} - {len(fc)} fifteen-minute blocks ahead.")

peak_mw = fc["pred_mw"].max()
peak_ts = fc.loc[fc["pred_mw"].idxmax(), "ts"]
total_mwh = fc["pred_mw"].sum() * 0.25
curtail_mwh = np.clip(fc["pred_mw"] - baseline_mw - battery_mw, 0, None).sum() * 0.25

c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak expected", f"{peak_mw:.1f} MW", f"at {peak_ts:%H:%M}")
c2.metric("Total yield", f"{total_mwh:,.0f} MWh", f"over {horizon}h")
c3.metric("Curtailment risk", f"{curtail_mwh:,.1f} MWh")
c4.metric("Demand baseline", f"{baseline_mw:.1f} MW")

# --- chart -----------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=fc["ts"], y=fc["upper_mw"], line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc["ts"], y=fc["lower_mw"], fill="tonexty",
                         fillcolor="rgba(255,166,0,0.18)", line=dict(width=0),
                         name="Confidence range"))
fig.add_trace(go.Scatter(x=fc["ts"], y=fc["pred_mw"], name="Forecast solar (MW)",
                         line=dict(color="#FFA600", width=2.5)))
fig.add_hline(y=baseline_mw, line=dict(color="#4C78A8", dash="dash"),
              annotation_text="Grid demand baseline")
fig.update_layout(height=430, hovermode="x unified",
                  margin=dict(l=10, r=10, t=30, b=10),
                  yaxis_title="Power (MW)", xaxis_title="")
st.plotly_chart(fig, use_container_width=True)

# --- grid action centre ----------------------------------------------------
st.subheader("Grid Action Centre")
recs = make_recommendations(fc, baseline_mw, battery_mw)
if not recs:
    st.info("No action required. Forecast output tracks the demand baseline.")
colour = {"CRITICAL": "#D62728", "WARNING": "#FF7F0E", "INFO": "#2CA02C"}
for sev, when, msg, action in recs:
    st.markdown(
        f"<div style='border-left:5px solid {colour[sev]};background:rgba(128,128,128,0.08);"
        f"padding:10px 14px;margin-bottom:8px;border-radius:4px;'>"
        f"<b style='color:{colour[sev]}'>{sev}</b> &nbsp; <b>{when:%a %d %b, %H:%M}</b>"
        f"<br>{msg}<br><b>Action: {action}</b></div>",
        unsafe_allow_html=True)

# --- proof panel -----------------------------------------------------------
st.subheader("Model validation: predicted vs actual")
st.caption("Days the model never saw during training. This is the evidence the "
           "forecast can be trusted.")
bt_path = os.path.join(OUT_DIR, "backtest.csv")
if os.path.exists(bt_path):
    bt = pd.read_csv(bt_path, parse_dates=["DATE_TIME"])
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=bt["DATE_TIME"], y=bt["ac_power_kw"] / 1000,
                            name="Actual", line=dict(color="#4C78A8", width=2)))
    f2.add_trace(go.Scatter(x=bt["DATE_TIME"], y=bt["predicted_kw"] / 1000,
                            name="Predicted", line=dict(color="#FFA600", width=2, dash="dot")))
    f2.update_layout(height=330, hovermode="x unified",
                     margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Power (MW)")
    st.plotly_chart(f2, use_container_width=True)
    st.caption(f"Trained on {info['trained_rows']:,} fifteen-minute records from "
               f"{info['date_range'][0][:10]} to {info['date_range'][1][:10]}.")
