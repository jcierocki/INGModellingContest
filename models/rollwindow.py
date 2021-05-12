import pandas as pd
import numpy as np
from multiprocess import Pool
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import time

TRAIN_LEN = 282
VAL_LEN = 24
HORIZON = 12


def ets_forecast(y, horizon):
    return SimpleExpSmoothing(y, initialization_method='heuristic').fit().forecast(horizon)[-1]


def rolling_window(y, func, window, horizon, num_threads=7):
    n_steps = len(y) - window - horizon
    start_idx = np.arange(window)
    end_idx = start_idx + window

    with Pool(num_threads) as pool:
        forecasts = pool.map(
            lambda idx_range: func(y[idx_range[0]:idx_range[1]], horizon),
            zip(start_idx[:n_steps], end_idx[:n_steps])
        )
        output = [x for x in forecasts]

    return output


df = pd.read_csv("../data/raw_data.csv").rename(
    columns={"DATE": "date", "Hard coal consumption per capita [tones]": "consumption"})
df.date = pd.to_datetime(df.date).dt.date
ts = df.consumption.values

start = time.perf_counter()
tt = rolling_window(ts, ets_forecast, window=TRAIN_LEN, horizon=HORIZON)
elapsed_time = time.perf_counter() - start
print(f"Elapsed time: {elapsed_time:0.4f} seconds")

start = time.perf_counter()
tt2 = rolling_window(ts, ets_forecast, window=TRAIN_LEN, horizon=HORIZON, num_threads=1)
elapsed_time = time.perf_counter() - start
print(f"Elapsed time: {elapsed_time:0.4f} seconds")

