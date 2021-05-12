import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.statespace.sarimax import *
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from hyperopt.pyll.base import scope
from numba import set_num_threads
import time

TRAIN_LEN = 282
VAL_LEN = 24
HORIZON = 12

df = pd.read_csv("../data/raw_data.csv").rename(
    columns={"DATE": "date", "Hard coal consumption per capita [tones]": "consumption"})
df.date = pd.to_datetime(df.date).dt.date

# ts = pd.Series(df.consumption.values, df.date)
ts = df.consumption.squeeze()

ts_train = ts.iloc[:TRAIN_LEN]
ts_val = ts.iloc[TRAIN_LEN:TRAIN_LEN + VAL_LEN]
ts_test = ts.iloc[TRAIN_LEN + VAL_LEN:]

ts_train_val = ts.iloc[:TRAIN_LEN + VAL_LEN - HORIZON]
ts_rolled = ts_train_val.rolling(window=TRAIN_LEN)
set_num_threads(7)

start = time.perf_counter()
roll_forecasts = ts_rolled.apply(
    func=lambda x: SARIMAX(x, order=(2, 0, 0), seasonal_order=(1, 0, 0, 12), trend='ct').fit(method='lbfgs').forecast(steps=12)[-1],
    raw=True,
    engine='numba',
    engine_kwargs={'nopython': False, 'nogil': True, 'parallel': True}
)
elapsed_time = time.perf_counter() - start
print(f"Elapsed time: {elapsed_time:0.4f} seconds")