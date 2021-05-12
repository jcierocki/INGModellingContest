import pandas as pd
import numpy as np
from multiprocess import Pool
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TRAIN_LEN = 282
VAL_LEN = 24
HORIZON = 12


def ets_forecast(y, horizon):
    return SimpleExpSmoothing(y, initialization_method='heuristic').fit().forecast(horizon)


def model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_true, y_pred)
        # kolejne do dopisania wedle uznania
    }


def eval_forecast_calibrate(y, func, metric_fun, horizon, idx_start, idx_end):
    y_train = y[idx_start:idx_end]
    y_val = y[idx_end:idx_end+horizon]

    y_pred = func(y_train, horizon)

    return metric_fun(y_val, y_pred)


def eval_forecast(y, func, horizon, idx_start, idx_end):
    y_train = y[idx_start:idx_end]
    y_val = y[idx_end:idx_end + horizon]

    y_pred = func(y_train, horizon)
    metrics = model_metrics(y_val, y_pred)

    return {'metrics': metrics, 'forecast': y_pred}


def rolling_window(y, func, window, horizon, num_threads=7):
    n_steps = len(y) - window - horizon
    idx_start = np.arange(window)
    idx_end = idx_start + window

    with Pool(num_threads) as pool:
        forecasts = pool.map(
            lambda idx_range: eval_forecast_calibrate(y, func, mean_squared_error, horizon, *idx_range),
            zip(idx_start[:n_steps], idx_end[:n_steps])
        )
        output = np.sum([x for x in forecasts])

    return output


df = pd.read_csv("../data/raw_data.csv").rename(
    columns={"DATE": "date", "Hard coal consumption per capita [tones]": "consumption"})
df.date = pd.to_datetime(df.date).dt.date
ts = df.consumption.values
ts_train_val = ts[:TRAIN_LEN+VAL_LEN]

start = time.perf_counter()
tt = rolling_window(ts_train_val, ets_forecast, window=TRAIN_LEN, horizon=HORIZON)
elapsed_time = time.perf_counter() - start
print(f"Elapsed time: {elapsed_time:0.4f} seconds")

start = time.perf_counter()
tt2 = rolling_window(ts_train_val, ets_forecast, window=TRAIN_LEN, horizon=HORIZON, num_threads=1)
elapsed_time = time.perf_counter() - start
print(f"Elapsed time: {elapsed_time:0.4f} seconds")