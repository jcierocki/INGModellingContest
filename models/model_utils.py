import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fbprophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
# from darts.metrics.metrics import mae, mase, mape, smape, rmse
# from darts.models.exponential_smoothing import ExponentialSmoothing
from tbats import TBATS
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model

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


def evaluate_calibration(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred)
    }


def freq_convert(freq_str):
    if freq_str == 'M':
        return 12
    elif freq_str == 'W':
        return 7
    elif freq_str == 'D':
        return None  # TODO check it
    else:
        raise AttributeError


def prepare_data_prophet(train, val, col, exog_col=None):
    colnames = col + exog_col

    train_data = train[colnames].copy()
    train_data["ds"] = train_data.index
    train_data = train_data.rename(columns={col: 'y'})

    val_data = val[colnames].copy()
    val_data["ds"] = val_data.index
    val_data = val_data.rename(columns={col: 'y'})

    return train_data, val_data


def prophet(train, val, col, exog_col=None, freq='M', eval_fun=evaluate):
    train_data, val_data = prepare_data_prophet(train, val, col)

    m = Prophet()
    if exog_col is not None:
        for c in exog_col:
            m.add_regressor(name=c)

    m.fit(train_data)

    future = m.make_future_dataframe(periods=val_data.shape[0], freq=freq)
    forecast = m.predict(future).tail(val_data.shape[0])["yhat"].values
    return eval_fun(val_data["y"].values, forecast), forecast


def neural_prophet(train, val, col, exog_col=None, freq='M', eval_fun=evaluate):
    train_data, val_data = prepare_data_prophet(train, val, col)

    m = NeuralProphet()
    if exog_col is not None:
        for c in exog_col:
            m.add_lagged_regressor(name=c)

    m.fit(train_data, freq=freq)

    future = m.make_future_dataframe(train_data, periods=val_data.shape[0])
    forecast = m.predict(future)["yhat1"].values
    return eval_fun(val_data["y"].values, forecast), forecast


def tbats(
        train,
        val,
        col="org",
        exog_col=None,
        freq='M',
        eval_fun=evaluate,
        use_box_cox=False,
        use_trend=True,
        use_damped_trend=False,
        use_arma_errors=True
):
    train_data = train[col].copy()
    val_data = val[col].copy()

    seasonal_periods = freq_convert(freq)

    estimator = TBATS(
        seasonal_periods=seasonal_periods,
        use_box_cox=use_box_cox,
        use_trend=use_trend,
        use_damped_trend=use_damped_trend,
        use_arma_errors=use_arma_errors
    )

    fitted_model = estimator.fit(train_data.values)

    forecast = fitted_model.forecast(steps=val_data.shape[0])
    return eval_fun(val_data, forecast), forecast


def sarimax(
        train,
        val,
        col="org",
        exog_col=None,
        eval_fun=evaluate,
        p=2,
        d=1,
        q=2,
        trend="c",
        P=0,
        D=0,
        Q=0,
        freq='M',
        use_boxcox=True,
):
    train_data = train[col].copy()
    train_data_exog = train[exog_col].copy()
    val_data = val[col].copy()

    s = freq_convert(freq)

    mod = sm.tsa.statespace.SARIMAX(
        endog=train_data,
        exog=train_data_exog,  # TODO check if numpy needed
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        trend=trend,
        use_boxcox=use_boxcox,
        initialization_method="estimated",
    )
    res = mod.fit(method='powell', maxiter=1000)

    forecast = res.forecast(steps=val_data.shape[0])
    return eval_fun(val_data, forecast), forecast


def ets(
        train,
        val,
        col="org",
        exog_col=None,
        eval_fun=evaluate,
        freq='M',
        error='add',
        trend='add',
        damped_trend=False,
        seasonal=None,
        epsilon=0
):
    train_data = train[col].copy() + epsilon
    val_data = val[col].copy() + epsilon

    seasonal_periods = freq_convert(freq)

    mod = ETSModel(
        train_data,
        error=error,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    res = mod.fit()

    forecast = res.forecast(steps=val_data.shape[0]) - epsilon
    return eval_fun(val_data, forecast), forecast


def var(train, val, col, exog_col=None, eval_fun=evaluate, freq='M', max_lags=None, trend='c'):
    colnames = col + exog_col

    train_data = train[colnames].copy()
    val_data = val[colnames].copy()

    # s = freq_convert(freq)

    mod = VAR(endog=train_data, freq=freq)  # TODO check freq arg format
    res = mod.fit(maxlags=max_lags, trend=trend)

    forecast = res.forecast(steps=val_data.shape[0], y=train_data)  # TODO check if y arg filled properly
    return eval_fun(val_data, forecast), forecast


def garch(train, val, col, exog_col=None, eval_fun=evaluate, freq='M', p=1, q=1, dist="Normal"):
    train_data = train[col].copy()
    train_data_exog = train[exog_col].copy()
    val_data = val[col].copy()

    mod = arch_model(
        y=train_data,
        # x=train_data_exog,  # TODO include exog regressors
        mean="Constant",
        lags=0,
        vol="Garch",
        p=p,
        o=0,
        q=q,
        power=2.0,
        dist=dist
    )
    res = mod.fit(options={'maxiter': 1000})

    forecast = res.forecast(horizon=val_data.shape[0])
    return eval_fun(val_data, forecast), forecast
