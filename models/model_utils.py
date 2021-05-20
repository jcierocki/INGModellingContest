import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fbprophet import Prophet
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
# from darts.metrics.metrics import mae, mase, mape, smape, rmse
# from darts.models.exponential_smoothing import ExponentialSmoothing
from tbats import TBATS
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
from sklearn.linear_model import ElasticNet

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


def mape(y_true, y_pred, eps=.001):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def smape(y_true, y_pred, eps=.001):
    return (
            100
            / len(y_true)
            * np.sum(2 * np.abs(y_pred - y_true) / (eps + np.abs(y_true) + np.abs(y_pred)))
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
    if freq_str == 'M' or freq_str == 'MS':
        return 12
    elif freq_str == 'W':
        return 7
    elif freq_str == 'D':
        return None  # TODO check it
    else:
        raise AttributeError


def prepare_data_prophet(train, val, col, exog_col=None):
    colnames = [col]
    if exog_col is not None:
        colnames += exog_col

    train_data = train[colnames].copy().rename(columns={col: 'y'})
    train_data["ds"] = train.index

    val_data = val[colnames].copy().rename(columns={col: 'y'})
    val_data["ds"] = val.index

    return train_data, val_data


# def split_exog(df, cols, horizon):
#     if cols is not None:
#         exog_data = df[cols].copy()
#         split_idx = exog_data.shape[0] - horizon
#         return exog_data.iloc[:split_idx, :], exog_data.iloc[split_idx:, :]
#     else:
#         return None, None


def prophet(train, val, col, exog_col=None, freq='MS', eval_fun=evaluate):
    train_data, val_data = prepare_data_prophet(train, val, col, exog_col)

    m = Prophet()
    if exog_col is not None:
        for c in exog_col:
            m.add_regressor(name=c)

    m.fit(train_data)

    future = m.make_future_dataframe(periods=val_data.shape[0], freq=freq)
    if exog_col is not None:
        for c in exog_col:
            future[c] = np.concatenate(
                (train_data[c].to_numpy(), val_data[c].to_numpy())
            )

    forecast = m.predict(future).tail(val_data.shape[0])["yhat"].values
    return eval_fun(val_data["y"].values, forecast), forecast


# def neural_prophet(train, val, col, exog_col=None, freq='M', eval_fun=evaluate, ar_lag=2, ar_hidden_layers=1):
#     train_data, val_data = prepare_data_prophet(train, val, col, exog_col)
#
#     m = NeuralProphet(n_forecasts=12, n_lags=ar_lag, num_hidden_layers=ar_hidden_layers)
#     if exog_col is not None:
#         for c in exog_col:
#             m = m.add_future_regressor(name=c)
#
#     if freq == 'M':  # TODO temporary
#         freq = 'MS'
#
#     m.fit(train_data, freq=freq)
#
#     future = m.make_future_dataframe(df=train_data, regressors_df=train_data[exog_col])
#
#     # print(future.shape[0])
#     # if exog_col is not None:
#     #     for c in exog_col:
#     #         future[c] = np.concatenate(
#     #             (train_data[c].to_numpy(), val_data[c].to_numpy())
#     #         )
#
#     forecast = m.predict(future)["yhat1"].values
#
#     return None, forecast
#     # return eval_fun(val_data["y"].values, forecast), forecast


def tbats(
        train,
        val,
        col="org",
        exog_col=None,
        freq='MS',
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
        seasonal_periods=[seasonal_periods],
        use_box_cox=use_box_cox,
        use_trend=use_trend,
        use_damped_trend=use_damped_trend,
        use_arma_errors=use_arma_errors,
        n_jobs=1
    )

    fitted_model = estimator.fit(train_data.values)

    forecast = fitted_model.forecast(steps=val_data.shape[0])
    return eval_fun(val_data, forecast), forecast


def sarimax(
        train,
        val,
        col,
        exog_col=None,
        freq='MS',
        eval_fun=evaluate,
        p=1,
        d=0,
        q=1,
        trend="c",
        P=0,
        D=0,
        Q=0,
        use_boxcox=True,
):
    train_data = train.copy()
    val_data = val.copy()

    xreg = None
    xreg_fcst = None
    if exog_col is not None:
        xreg = train_data[list(exog_col)]
        xreg_fcst = val_data[list(exog_col)]

    s = freq_convert(freq)

    mod = sm.tsa.statespace.SARIMAX(
        endog=train_data[col],
        exog=xreg,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        trend=trend,
        use_boxcox=use_boxcox,
        initialization_method="estimated",
        freq=freq
    )
    res = mod.fit(method='powell', maxiter=1000, disp=False)

    forecast = res.forecast(steps=val_data.shape[0], exog=xreg_fcst)
    return eval_fun(val_data[col], forecast), forecast


def ets(
        train,
        val,
        col="org",
        exog_col=None,
        freq='MS',
        eval_fun=evaluate,
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
        freq=freq,
        initialization_method="estimated",
    )
    res = mod.fit()

    forecast = res.forecast(steps=val_data.shape[0]) - epsilon
    return eval_fun(val_data, forecast), forecast


def var(train, val, col, exog_col=None, freq='MS', eval_fun=evaluate, max_lags=None, trend='c'):
    if exog_col is None:
        raise Exception

    colnames = [col] + list(exog_col)

    train_data = train[colnames].copy()
    val_data = val[colnames].copy()

    mod = VAR(endog=train_data, freq=freq)
    res = mod.fit(maxlags=max_lags, trend=trend)

    forecast = res.forecast(steps=val_data.shape[0], y=train_data.to_numpy())
    return eval_fun(val_data, forecast), forecast


def garch(train, val, col, exog_col=None, freq='MS', eval_fun=evaluate, ar_lag=2, p=1, q=1, dist="Normal"):
    train_data = train.copy()
    val_data = val.copy()

    xreg = None
    xreg_fcst = None
    if exog_col is not None:
        xreg = train_data[list(exog_col)]
        xreg_fcst = val_data[list(exog_col)].to_dict(orient='list')

    mod = arch_model(
        y=train_data[col],
        x=xreg,
        lags=ar_lag,
        mean="ARX",
        vol="Garch",
        p=p,
        o=0,
        q=q,
        power=2.0,
        dist=dist,
        rescale=True
    )
    res = mod.fit(options={'maxiter': 1000}, disp='off')

    forecast_arch = res.forecast(horizon=val_data.shape[0], x=xreg_fcst, reindex=True)
    forecast = pd.Series(forecast_arch.mean.iloc[-1, :])
    forecast.index = val_data.index

    return eval_fun(val_data[col], forecast), forecast


def elastic_net(train, val, resid, exog_col, col=None, eval_fun=evaluate, alpha=1.0, l1=0.5):
    if resid is None or exog_col is None:
        raise Exception

    train_data = train[list(exog_col)].copy()
    val_data = val[list(exog_col)].copy()

    mod = ElasticNet(alpha=alpha, l1_ratio=l1)
    res = mod.fit(X=train_data.to_numpy(), y=resid.to_numpy())
    forecast = res.predict(val_data)

    metrics = None
    if col is not None:
        metrics = eval_fun(val[col].values, forecast)

    return metrics, forecast




