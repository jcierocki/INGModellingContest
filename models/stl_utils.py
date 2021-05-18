import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def stl_ets(train, val, col='org', period=12, seasonal=7, trend=23, low_pass=13, seasonal_deg=0, trend_deg=0,
            low_pass_deg=0):
    train_data = train[col].copy()
    train_data.index = pd.DatetimeIndex(train_data.index).to_period('M')  # prowizorka
    val_data = val[col].copy()

    mod = STLForecast(
        endog=train_data,
        model=SimpleExpSmoothing,
        model_kwargs={'initialization_method': 'estimated'},
        period=period,
        seasonal=seasonal,
        trend=trend,
        low_pass=low_pass,
        seasonal_deg=seasonal_deg,
        trend_deg=trend_deg,
        low_pass_deg=low_pass_deg,
        robust=True
    )

    res = mod.fit()
    forecast = res.forecast(steps=val_data.shape[0])

    # prowizorka
    return {'mae': mean_absolute_error(np.squeeze(val_data), np.squeeze(forecast))}, forecast
