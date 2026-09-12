"""
=============================================================================
 STEP 1 of 2 — TRAIN THE MODEL
=============================================================================

 What this does, in plain English:
   1. Reads the two Kaggle CSV files.
   2. Adds up all 22 inverters to get the whole plant's output.
   3. Joins the power records to the weather records by timestamp.
   4. Teaches XGBoost the pattern: "this much sun + this hot = this much power".
   5. Tests it on days it has never seen, and prints how accurate it was.
   6. Saves the trained model so the dashboard can use it.

 How to run it:
       python train_model.py

 Takes about 30 seconds. You only run this ONCE.
=============================================================================
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# SETTINGS — the only things you might want to change
# ---------------------------------------------------------------------------
DATA_DIR = "data"
GENERATION_CSV = "Plant_1_Generation_Data.csv"
WEATHER_CSV = "Plant_1_Weather_Sensor_Data.csv"

TIMEZONE = "Asia/Kolkata"
TEST_FRACTION = 0.20          # last 20% of days are the exam the model never saw
OUT_DIR = "outputs"


# ---------------------------------------------------------------------------
# Reading dates safely.
#
# The two files write dates differently. One is "15-05-2020 08:30" (day first),
# the other is "2020-05-15 08:30:00" (year first). If we let the computer guess,
# it can read 05-06-2020 as 5th June instead of 6th May -- with NO error message.
# Everything would still run and every number would be wrong.
#
# So we try each known format explicitly and keep the one that works.
# ---------------------------------------------------------------------------
DATE_FORMATS = [
    "%d-%m-%Y %H:%M",       # 15-05-2020 08:30   (generation file)
    "%Y-%m-%d %H:%M:%S",    # 2020-05-15 08:30:00 (weather file)
    "%d-%m-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M",
]


def parse_dates(series, filename):
    """Try each known date format. Refuse to guess."""
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.99:
            print(f"   {filename}: read dates as '{fmt}'  OK")
            return parsed
    sys.exit(
        f"\nERROR: could not read the dates in {filename}.\n"
        f"First value looks like: {series.iloc[0]!r}\n"
        f"Add its format to the DATE_FORMATS list near the top of this file."
    )


def load_and_merge():
    gen_path = os.path.join(DATA_DIR, GENERATION_CSV)
    wx_path = os.path.join(DATA_DIR, WEATHER_CSV)

    for p in (gen_path, wx_path):
        if not os.path.exists(p):
            sys.exit(
                f"\nERROR: cannot find {p}\n"
                f"Download the CSVs from Kaggle and put them in the '{DATA_DIR}' folder."
            )

    print("\n[1/5] Reading the CSV files...")
    gen = pd.read_csv(gen_path)
    wx = pd.read_csv(wx_path)
    gen["DATE_TIME"] = parse_dates(gen["DATE_TIME"], GENERATION_CSV)
    wx["DATE_TIME"] = parse_dates(wx["DATE_TIME"], WEATHER_CSV)

    print(f"   generation rows: {len(gen):,}  (22 inverters x each timestamp)")
    print(f"   weather rows   : {len(wx):,}")

    # --- add up the 22 inverters into one plant total ------------------------
    # We use AC_POWER, not DC_POWER. In the Plant 1 file the DC column is about
    # 10x the AC column -- a recording quirk in the data, not real physics.
    print("\n[2/5] Adding up the 22 inverters into one plant total...")
    plant = (
        gen.groupby("DATE_TIME", as_index=False)["AC_POWER"]
        .sum()
        .rename(columns={"AC_POWER": "ac_power_kw"})
    )
    print(f"   {len(plant):,} timestamps, peak {plant['ac_power_kw'].max():,.0f} kW")

    # --- join power to weather ----------------------------------------------
    print("\n[3/5] Matching power records to weather records...")
    wx_small = wx[["DATE_TIME", "AMBIENT_TEMPERATURE",
                   "MODULE_TEMPERATURE", "IRRADIATION"]]
    df = plant.merge(wx_small, on="DATE_TIME", how="inner")
    df = df.sort_values("DATE_TIME").drop_duplicates("DATE_TIME").reset_index(drop=True)

    if len(df) < 500:
        sys.exit(
            f"\nERROR: only {len(df)} rows matched between the two files.\n"
            f"This almost always means the dates were read wrongly. Check the "
            f"format messages printed above."
        )
    print(f"   {len(df):,} rows matched")
    print(f"   date range: {df['DATE_TIME'].min()}  ->  {df['DATE_TIME'].max()}")

    # Kaggle's IRRADIATION peaks near 1.2, so it is kW/m2. Convert to W/m2
    # so it matches what the weather service gives us later.
    df["irradiance_wm2"] = df["IRRADIATION"] * 1000.0
    df = df.rename(columns={
        "AMBIENT_TEMPERATURE": "ambient_temp_c",
        "MODULE_TEMPERATURE": "module_temp_c",
    })
    return df


# ---------------------------------------------------------------------------
# Sun position. This is pure astronomy -- like a calendar. It is exact for any
# minute of any year, so it is the most reliable information we have.
# ---------------------------------------------------------------------------
def solar_elevation(timestamps, lat, lon, tz_offset_hours=5.5):
    """Angle of the sun above the horizon, in degrees. Negative = night."""
    ts = pd.DatetimeIndex(timestamps)
    doy = ts.dayofyear.values
    hour = ts.hour.values + ts.minute.values / 60.0

    decl = np.radians(23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.0)))
    b = np.radians(360.0 * (doy - 81) / 364.0)
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)

    solar_time = hour + (4.0 * (lon - 15.0 * tz_offset_hours) + eot) / 60.0
    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    lat_r = np.radians(lat)

    sin_elev = (np.sin(lat_r) * np.sin(decl)
                + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))


def add_features(df, lat, lon):
    """Build the columns the model learns from."""
    ts = pd.DatetimeIndex(df["DATE_TIME"])

    # Time of day, encoded as a circle. If we used plain numbers, the computer
    # would think 23:45 and 00:00 are 24 hours apart instead of 15 minutes.
    slot = ts.hour * 4 + ts.minute // 15
    df["tod_sin"] = np.sin(2 * np.pi * slot / 96)
    df["tod_cos"] = np.cos(2 * np.pi * slot / 96)
    df["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365)

    df["solar_elevation"] = solar_elevation(ts, lat, lon)
    df["is_daylight"] = df["solar_elevation"] > 0

    # Rolling sunshine averages: was it sunny over the last hour / three hours?
    # These are legal because we will have forecast sunshine for every future
    # timestamp too. (Rolling averages of POWER would be cheating -- we will not
    # know tomorrow's power when making tomorrow's forecast.)
    df["irr_roll_1h"] = df["irradiance_wm2"].rolling(4, min_periods=1).mean()
    df["irr_roll_3h"] = df["irradiance_wm2"].rolling(12, min_periods=1).mean()

    # How hot the panel is above ambient -- panels lose efficiency when hot.
    df["temp_diff"] = df["module_temp_c"] - df["ambient_temp_c"]
    df["irr_x_temp"] = df["irradiance_wm2"] * df["ambient_temp_c"] / 1000.0
    return df


FEATURES = [
    "irradiance_wm2", "ambient_temp_c", "module_temp_c",
    "irr_roll_1h", "irr_roll_3h", "temp_diff", "irr_x_temp",
    "solar_elevation", "tod_sin", "tod_cos", "doy_sin", "doy_cos",
]
TARGET = "ac_power_kw"


# ---------------------------------------------------------------------------
# Accuracy measures
# ---------------------------------------------------------------------------
def wape(actual, predicted):
    """Weighted Absolute Percentage Error. Lower is better. Under 10% is good."""
    total = np.abs(actual).sum()
    return np.nan if total == 0 else 100.0 * np.abs(actual - predicted).sum() / total


def main():
    lat, lon = 23.90, 71.10          # Charanka Solar Park, Gujarat (see README)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_and_merge()
    df = add_features(df, lat, lon)

    # --- split by TIME, never randomly -------------------------------------
    # The model is tested on the LAST days only. If we split randomly, the model
    # would see 10:00 and 10:30 of the same morning in training and testing, and
    # score brilliantly while having learned nothing useful.
    print("\n[4/5] Splitting into practice days and exam days...")
    cut_ts = df["DATE_TIME"].iloc[int(len(df) * (1 - TEST_FRACTION))].normalize()
    train = df[df["DATE_TIME"] < cut_ts]
    test = df[df["DATE_TIME"] >= cut_ts]
    print(f"   practice on: {train['DATE_TIME'].min().date()} -> {train['DATE_TIME'].max().date()}  ({len(train):,} rows)")
    print(f"   exam on    : {test['DATE_TIME'].min().date()} -> {test['DATE_TIME'].max().date()}  ({len(test):,} rows)")

    print("\n[5/5] Training the model...")
    model = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    model.fit(train[FEATURES], train[TARGET],
              eval_set=[(test[FEATURES], test[TARGET])], verbose=False)

    pred = model.predict(test[FEATURES])

    # Physical reality check: no negative power, and no power at night.
    pred = np.clip(pred, 0, None)
    pred[~test["is_daylight"].values] = 0.0

    actual = test[TARGET].values
    daylight = test["is_daylight"].values
    rated = float(np.percentile(df[TARGET], 99.5))

    print("\n" + "=" * 62)
    print(" RESULTS  (on days the model had never seen)")
    print("=" * 62)
    print(f" WAPE, daylight hours : {wape(actual[daylight], pred[daylight]):6.2f} %   <-- the honest number")
    print(f" WAPE, all 24 hours   : {wape(actual, pred):6.2f} %")
    print(f" MAE  (average miss)  : {np.mean(np.abs(actual - pred)):9.1f} kW")
    print(f" RMSE                 : {np.sqrt(np.mean((actual - pred) ** 2)):9.1f} kW")
    print("=" * 62)

    print("\n What the model relies on most:")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_),
                            key=lambda x: -x[1])[:5]:
        print(f"   {name:20s} {'#' * int(imp * 50)} {imp:.3f}")

    # Save the underlying booster, not the sklearn wrapper: on some
    # xgboost/scikit-learn version pairs wrapper.save_model() crashes.
    model.get_booster().save_model(os.path.join(OUT_DIR, "solar_model.json"))
    test_out = test[["DATE_TIME", TARGET, "is_daylight"]].copy()
    test_out["predicted_kw"] = pred
    test_out.to_csv(os.path.join(OUT_DIR, "backtest.csv"), index=False)

    with open(os.path.join(OUT_DIR, "model_info.json"), "w") as f:
        json.dump({
            "features": FEATURES,
            "rated_capacity_kw": rated,
            "latitude": lat, "longitude": lon, "timezone": TIMEZONE,
            "wape_daylight": float(wape(actual[daylight], pred[daylight])),
            "wape_all": float(wape(actual, pred)),
            "mae_kw": float(np.mean(np.abs(actual - pred))),
            "trained_rows": int(len(train)),
            "date_range": [str(df["DATE_TIME"].min()), str(df["DATE_TIME"].max())],
        }, f, indent=2)

    print(f"\n Saved to '{OUT_DIR}/'. Now run:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
