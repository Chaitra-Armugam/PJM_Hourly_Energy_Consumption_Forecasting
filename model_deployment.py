"""Backend for the PJM hourly energy-consumption forecasting deployment."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "PJM_selected_deployment_model.joblib"
HISTORY_PATH = BASE_DIR / "PJMW_MW_cleaned_hourly_history.csv"
METADATA_PATH = BASE_DIR / "PJM_deployment_metadata.json"
FEATURES_PATH = BASE_DIR / "PJM_tree_feature_columns.json"

# ---------------------------------------------------------------------------------------------
# Feature engineering — identical to the notebook (Section 3 of Week 2) so training and serving match.
# ---------------------------------------------------------------------------------------------
HORIZON_HOURS = 720                  # longest forecast horizon: 30 days x 24 hours
YEAR_LAG = 364 * 24                  # 8,736 h = exactly 52 weeks -> same weekday and hour one year earlier
MIN_ORIGIN_POSITION = YEAR_LAG + HORIZON_HOURS + 200   # earliest usable forecast origin inside the history
US_HOLIDAYS = USFederalHolidayCalendar().holidays(start='2000-01-01', end='2035-12-31')

CALENDAR_FEATURES = [
    'hour', 'day_of_week', 'day_of_year', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'year_sin', 'year_cos',
    'is_holiday', 'is_holiday_eve', 'is_holiday_after',
]
ANCHORED_FEATURES = [
    'steps_ahead', 'last_value', 'mean_24', 'mean_168', 'mean_720',
    'anchor_24', 'anchor_168', 'anchor_168_avg4', 'anchor_24_avg7',
    'last_year_same_hour', 'last_year_smooth', 'growth_720',
]
FEATURES = CALENDAR_FEATURES + ANCHORED_FEATURES


def calendar_features(target_index):
    """Calendar / holiday features of the timestamps being predicted (known in advance for any future date)."""
    target_index = pd.DatetimeIndex(target_index)
    hour, dow, doy = target_index.hour, target_index.dayofweek, target_index.dayofyear
    day = target_index.normalize()
    return pd.DataFrame({
        'hour': hour, 'day_of_week': dow, 'day_of_year': doy, 'month': target_index.month,
        'is_weekend': (dow >= 5).astype(int),
        'hour_sin': np.sin(2*np.pi*hour/24), 'hour_cos': np.cos(2*np.pi*hour/24),
        'dow_sin': np.sin(2*np.pi*dow/7), 'dow_cos': np.cos(2*np.pi*dow/7),
        'year_sin': np.sin(2*np.pi*doy/365.25), 'year_cos': np.cos(2*np.pi*doy/365.25),
        'is_holiday': day.isin(US_HOLIDAYS).astype(int),
        'is_holiday_eve': (day + pd.Timedelta(days=1)).isin(US_HOLIDAYS).astype(int),
        'is_holiday_after': (day - pd.Timedelta(days=1)).isin(US_HOLIDAYS).astype(int),
    }, index=target_index)[CALENDAR_FEATURES]


def anchored_features(values, origin_pos, steps):
    """History-anchored features. `origin_pos` is the position of the LAST KNOWN observation and `steps` is how many
    hours ahead the target is (1..720). Every value used lies at or before the origin, so there is no leakage."""
    p = np.asarray(origin_pos, dtype=int)
    k = np.asarray(steps, dtype=int)
    q = p + k                                            # position of the target hour
    cs = np.concatenate([[0.0], np.cumsum(values)])
    mean_last = lambda n, pos: (cs[pos + 1] - cs[pos + 1 - n]) / n    # mean of the n hours ending at `pos`
    c24 = np.ceil(k / 24).astype(int)                    # whole days needed to reach back before the origin
    c168 = np.ceil(k / 168).astype(int)                  # whole weeks needed to reach back before the origin
    return pd.DataFrame({
        'steps_ahead': k,
        'last_value': values[p],
        'mean_24': mean_last(24, p), 'mean_168': mean_last(168, p), 'mean_720': mean_last(720, p),
        'anchor_24': values[q - 24*c24],                 # same hour of day, most recent known day
        'anchor_168': values[q - 168*c168],              # same hour of week, most recent known week
        'anchor_168_avg4': np.mean([values[q - 168*(c168 + j)] for j in range(4)], axis=0),
        'anchor_24_avg7': np.mean([values[q - 24*(c24 + j)] for j in range(7)], axis=0),
        'last_year_same_hour': values[q - YEAR_LAG],     # same weekday/hour, 52 weeks earlier
        'last_year_smooth': np.mean([values[q - YEAR_LAG + 24*d] for d in range(-3, 4)], axis=0),
        'growth_720': mean_last(720, p) / mean_last(720, p - YEAR_LAG),   # recent level vs the same window a year ago
    })[ANCHORED_FEATURES]


def build_features(values, target_index, origin_pos, steps):
    """Full feature matrix (calendar + history-anchored) in the exact column order used for training."""
    calendar = calendar_features(target_index).reset_index(drop=True)
    anchored = anchored_features(values, origin_pos, steps)
    return pd.concat([calendar, anchored], axis=1)[FEATURES]


class AverageEnsemble:
    """Simple average of several fitted regressors. Kept here (not in the notebook) so that the saved
    .joblib file can always be loaded by the Streamlit app, which imports this module."""
    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([model.predict(X) for model in self.models], axis=0)

    @property
    def feature_importances_(self):
        return np.mean([m.feature_importances_ / m.feature_importances_.sum() for m in self.models], axis=0)


def load_metadata():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

def load_history():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing {HISTORY_PATH.name}; keep it beside app.py.")
    frame = pd.read_csv(HISTORY_PATH, parse_dates=["Datetime"])
    if not {"Datetime", "PJMW_MW"}.issubset(frame.columns):
        raise ValueError("History CSV must contain Datetime and PJMW_MW columns.")
    frame = frame.dropna(subset=["Datetime", "PJMW_MW"]).sort_values("Datetime")
    series = (frame.drop_duplicates("Datetime", keep="last")
              .set_index("Datetime")["PJMW_MW"])
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series = series.resample("h").mean().interpolate(
        method="time", limit=2, limit_area="inside"
    ).dropna()
    if len(series) < YEAR_LAG + HORIZON_HOURS + 200:
        raise ValueError("History must contain at least 1.1 years of hourly observations.")
    return series

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name}; keep it beside app.py.")
    return joblib.load(MODEL_PATH)

def forecast_hours(model, history, hours=720):
    """Generate hourly predictions for the next `hours` hours with the notebook's feature order.
    Every hour is predicted directly from the last observed history (no recursive error build-up)."""
    hours = int(hours)
    if not 1 <= hours <= HORIZON_HOURS:
        raise ValueError("Forecast horizon must be between 1 and 720 hours (30 days).")
    history = history.copy().sort_index()
    values = history.to_numpy(dtype=float)
    future_index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=hours, freq="h")
    origin = np.full(hours, len(values) - 1)
    steps = np.arange(1, hours + 1)
    features = build_features(values, future_index, origin, steps)
    predictions = np.maximum(0.0, np.asarray(model.predict(features), dtype=float))
    return pd.DataFrame({"Datetime": future_index, "Predicted_MW": predictions})

def export_forecast(model, history, hours=720, output_path=None):
    result = forecast_hours(model, history, hours)
    output = Path(output_path) if output_path else BASE_DIR / "PJM_forecast_output.csv"
    result.to_csv(output, index=False)
    return result, output

if __name__ == "__main__":
    model, history = load_model(), load_history()
    forecast, path = export_forecast(model, history, 720)
    print(f"Saved {len(forecast)} hourly predictions to {path}")
    print(f"Forecast period: {forecast.Datetime.min()} to {forecast.Datetime.max()}")
