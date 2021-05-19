import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hyperopt_utils import single_experiment
from hyperopt import fmin, tpe, space_eval, hp
from hyperopt.pyll.base import scope
from datetime import datetime


# TODO bind freq and period args
def stl_ets(train, val, col, freq='M', period=12, seasonal=7, trend=23, low_pass=13, seasonal_deg=0, trend_deg=0,
            low_pass_deg=0):
    train_data = train[col].copy()
    train_data.index = pd.DatetimeIndex(train_data.index).to_period(freq)
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

    return {'mae': mean_absolute_error(np.squeeze(val_data), np.squeeze(forecast))}, forecast


def optimize_stl(df, col, train_end, horizon, freq='month', max_evals=200):
    def objective(args):
        return single_experiment(
            df,
            frequency=freq,
            train_end=train_end,
            forecast_column=col,
            forecast_horizon=horizon,
            forecast_metric="mae",
            forecast_function=stl_ets,
            **args
        )

    space = {
        'seasonal': hp.choice('seasonal', [i for i in range(7, 16) if i % 2 != 0]),
        'trend': hp.choice('trend', [i for i in range(12, 26) if i % 2 != 0]),
        'low_pass': hp.choice('low_pass', [i for i in range(12, 16) if i % 2 != 0]),
        'seasonal_deg': hp.choice('seasonal_deg', [0, 1]),
        'trend_deg': hp.choice('trend_deg', [0, 1]),
        'low_pass_deg': hp.choice('low_pass_deg', [0, 1])
    }

    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)

    return space_eval(space, best)


def filter_outliers(resid, q=(.03, .97)):
    qmin, qmax = np.quantile(resid, q=list(q))

    return (resid < qmin) | (resid > qmax)


def adf_test(ts, max_lag=12):
    stat, pval, _, _, _, _ = adfuller(ts, maxlag=max_lag)

    return f"""
        ADF test statistics: {stat}
        p-value: {str(pval)}
        H0: unit root exists => time series non-stationary
    """
