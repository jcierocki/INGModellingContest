import pandas as pd
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from fbprophet import Prophet
from neuralprophet import NeuralProphet
from tbats import TBATS, BATS

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss, bds
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from scipy.special import boxcox, inv_boxcox
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.api import ARIMA
import logging


# logg = logging.getLogger(__name__)
# fh = logging.FileHandler('logs.log')
# logg.addHandler(fh)

logger = logging.getLogger("fbprophet")
logger.propagate = False

logger = logging.getLogger("neuralprophet")
logger.propagate = False


def plot_prediction(train, val, preds, title="", future=False):
    prediction = pd.DataFrame()
    prediction["real"] = np.concatenate((train, val))
    prediction["yhat"] = np.concatenate((train, np.squeeze(preds)), axis=0)

    if future:
        prediction = pd.DataFrame()
        prediction["real"] = train
        prediction["yhat"] = train

    plt.figure(figsize=(10, 3))
    plt.plot(range(prediction.shape[0]), prediction["real"].values, label="real")
    plt.plot(
        range(train.shape[0], preds.shape[0] + train.shape[0]), preds, label="forecast"
    )
    plt.legend(loc="best")
    plt.title(title)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.01))) * 100


def smape(y_true, y_pred):
    return (
        100
        / len(y_true)
        * np.sum(2 * np.abs(y_pred - y_true) / (0.01 + np.abs(y_true) + np.abs(y_pred)))
    )


def evaluate(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    scores = {}
    scores["mae"] = mean_absolute_error(y_true, y_pred)
    scores["mse"] = mean_squared_error(y_true, y_pred)
    scores["r2"] = r2_score(y_true, y_pred)
    scores["mape"] = mape(y_true, y_pred)
    scores["smape"] = smape(y_true, y_pred)
    return scores


def prophet(train, val, col="org"):
    train_data = pd.DataFrame()
    train_data["y"] = train[col].copy()
    train_data["ds"] = train_data.index
    train_data.columns = ["y", "ds"]

    val_data = pd.DataFrame()
    val_data["y"] = val[col].copy()
    val_data["ds"] = val_data.index
    val_data.columns = ["y", "ds"]

    m = Prophet()
    m.fit(train_data)

    future = m.make_future_dataframe(periods=val_data.shape[0], freq="M")
    forecast = m.predict(future).tail(val_data.shape[0])["yhat"].values
    return evaluate(val_data["y"].values, forecast), forecast


def neural_prophet(train, val, col="org"):
    train_data = pd.DataFrame()
    train_data["y"] = train[col].copy()

    train_data["ds"] = train_data.index
    train_data.columns = ["y", "ds"]

    val_data = pd.DataFrame()
    val_data["y"] = val[col].copy()
    val_data["ds"] = val_data.index
    val_data.columns = ["y", "ds"]

    m = NeuralProphet()
    m.fit(train_data, freq="M")
    future = m.make_future_dataframe(train_data, periods=val_data.shape[0])
    forecast = m.predict(future)["yhat1"].values
    return evaluate(val_data["y"].values, forecast), forecast


def tbats(train, val, col="org"):
    train_data = train[col].copy()
    val_data = val[col].copy()

    estimator = TBATS(seasonal_periods=[12, 5])

    fitted_model = estimator.fit(train_data.values)

    forecast = fitted_model.forecast(steps=val_data.shape[0])
    return evaluate(val_data, forecast), forecast


def sarima(
    train,
    val,
    col="org",
    future=False,
    p=2,
    d=1,
    q=2,
    trend="c",
    P=0,
    D=0,
    Q=0,
    s=12,
    use_boxcox=True,
):
    train_data = train[col].copy()
    val_data = val[col].copy()
    mod = sm.tsa.statespace.SARIMAX(
        train_data,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        trend=trend,
        use_boxcox=use_boxcox,
        initialization_method="estimated",
    )
    res = mod.fit()

    forecast = res.forecast(steps=val_data.shape[0])
    if not future:
        return evaluate(val_data, forecast), forecast
    else:
        return evaluate(forecast, forecast), forecast


def ets(train, val, col="org", trend="mul", seasonal="mul", use_boxcox=False):
    epsilon = 0.001
    train_data = train[col].copy() + epsilon
    val_data = val[col].copy() + epsilon

    mod = ExponentialSmoothing(
        train_data,
        seasonal_periods=12,
        trend=trend,
        seasonal=seasonal,
        use_boxcox=use_boxcox,
        initialization_method="estimated",
    )
    res = mod.fit()

    forecast = res.forecast(steps=val_data.shape[0]) - epsilon
    return evaluate(val_data, forecast), forecast
