import numpy as np
import pandas as pd
import statsmodels.api as sm
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import pandas as pd
from darts.metrics import (
    mae,
    mse,
    rmse,
    rmsle,
    mape,
    smape,
    mase,
    ope,
    marre,
    r2_score,
    coefficient_of_variation,
)
from darts import TimeSeries
from arch import arch_model
from fbprophet import Prophet
from neuralprophet import NeuralProphet
import itertools
import statsmodels.api as sm
from tbats import TBATS
from darts.models import TCNModel, RNNModel, NBEATSModel, TransformerModel
from darts.dataprocessing.transformers import Scaler
import logging


###########################################################################################################################################################
# -------------------------------------HELEPRS-------------------------------------------------------------------------------------------------------------
###########################################################################################################################################################


def freq_convert(freq_str):
    if freq_str == "M":
        return 12
    elif freq_str == "W":
        return 7
    elif freq_str == "D":
        return None  # TODO check it
    else:
        raise AttributeError


def evaluate(y_true, y_pred, insample):
    scores = {}
    scores["mae"] = mae(y_true, y_pred)
    scores["mse"] = mse(y_true, y_pred)
    scores["r2"] = r2_score(y_true, y_pred)
    scores["mape"] = mape(y_true, y_pred)
    scores["smape"] = smape(y_true, y_pred)
    scores["mase"] = mase(y_true, y_pred, insample)
    return scores


###########################################################################################################################################################
# -------------------------------------BASE TRAINERS-------------------------------------------------------------------------------------------------------
###########################################################################################################################################################


class BaseTrainer(object):
    def __init__(
        self,
        train_data,
        val_data,
        col="org",
        eval_fun=evaluate,
        freq_str="M",
        exog_col=None,
        model_kwargs={},
        fit_kwargs={},
    ):
        self.train_data = train_data.copy()
        self.val_data = val_data.copy()
        self.train_y = self.train_data[col]
        self.val_y = self.val_data[col]
        self.col = col
        self.exog_col = exog_col
        self.eval_fun = eval_fun
        self.freq = freq_convert(freq_str)
        self.freq_str = freq_str
        self.model_kwargs = model_kwargs
        self.fit_kwargs = fit_kwargs
        self.forecast = None

    # to darts series format
    def to_series(self, series):
        return TimeSeries.from_series(series)

    def eval(self):
        if self.forecast is not None:
            self.forecast = pd.Series(np.squeeze(self.forecast))
            self.forecast.index = self.val_y.index
            forecast = self.to_series(self.forecast)
            return (
                self.eval_fun(
                    self.to_series(self.val_y), forecast, self.to_series(self.train_y)
                ),
                forecast,
            )

    def run(self):
        self.preprocess_data()
        self.train()
        self.predict()
        return self.eval()

    def preprocess_data(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


class BaseProphetTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_data(self):
        if self.exog_col is not None:
            colnames = [self.col] + self.exog_col
            train_data = self.train_data[colnames].copy()
            val_data = self.val_data[colnames].copy()

        else:
            colnames = self.col
            train_data = self.train_data[colnames].copy().to_frame()
            val_data = self.val_data[colnames].copy().to_frame()

        train_data["ds"] = train_data.index
        self.train_data = train_data.rename(columns={self.col: "y"})

        val_data["ds"] = val_data.index
        self.val_data = val_data.rename(columns={self.col: "y"})


class BaseDARTSTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_data(self):
        self.cov_tr = None
        self.cov_val = None
        if self.exog_col is not None:
            if len(self.exog_col) > 1:
                cov_tr = TimeSeries.from_dataframe(
                    self.train_data[self.exog_col].copy()
                )
                cov_val = TimeSeries.from_dataframe(self.val_data[self.exog_col].copy())
            else:
                cov_tr = self.to_series(self.train_data[self.exog_col].copy())
                cov_val = self.to_series(self.val_data[self.exog_col].copy())

            cov_scaler = Scaler()
            cov_scaler.fit(cov_tr)
            self.cov_tr = cov_scaler.transform(cov_tr)
            self.cov_val = cov_scaler.transform(cov_val)

        self.train_data = self.to_series(self.train_data[self.col])
        self.val_data = self.to_series(self.val_data[self.col])

        self.scaler = Scaler()
        self.scaler.fit(self.train_data)
        self.train_data = self.scaler.transform(self.train_data)
        self.val_data = self.scaler.transform(self.val_data)

    def predict(self):
        self.forecast = self.model.predict(
            self.val_y.shape[0], series=self.train_data, covariates=self.cov_tr
        )
        self.forecast = self.scaler.inverse_transform(self.forecast)._values.flatten()


###########################################################################################################################################################
# -------------------------------------Univariate  methods-------------------------------------------------------------------------------------------------
###########################################################################################################################################################


class ETSTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_data(self):
        self.train_data = self.train_data[self.col]
        self.val_data = self.val_data[self.col]

    def train(self):
        self.model = ETSModel(self.train_data, **self.model_kwargs)
        self.res = self.model.fit(**self.fit_kwargs)

    def predict(self):
        self.forecast = self.res.forecast(steps=self.val_data.shape[0])


class TBATSTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_data(self):
        self.train_data = self.train_data[self.col]
        self.val_data = self.val_data[self.col]

    def train(self):
        self.model = TBATS(**self.model_kwargs)
        self.res = self.model.fit(self.train_data.values, **self.fit_kwargs)

    def predict(self):
        self.forecast = self.res.forecast(steps=self.val_data.shape[0])


class ProphetTrainer(BaseProphetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.model = Prophet(**self.model_kwargs)
        if self.exog_col is not None:
            for c in self.exog_col:
                self.model.add_regressor(name=c)
        self.model.fit(self.train_data, **self.fit_kwargs)

    def predict(self):
        future = self.model.make_future_dataframe(
            periods=self.val_data.shape[0], freq=self.freq_str
        )
        self.forecast = (
            self.model.predict(future).tail(self.val_data.shape[0])["yhat"].values
        )


class NeuralProphetTrainer(BaseProphetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.model = NeuralProphet(**self.model_kwargs)
        if self.exog_col is not None:
            for c in self.exog_col:
                self.model.add_lagged_regressor(name=c)

        self.model.fit(self.train_data, freq=self.freq_str)

    def predict(self):
        future = self.model.make_future_dataframe(
            self.train_data, periods=self.val_data.shape[0]
        )
        self.forecast = self.model.predict(future)["yhat1"].values


class NBEATSTrainer(BaseDARTSTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.model = NBEATSModel(
            input_chunk_length=self.val_y.shape[0] * 3,
            output_chunk_length=self.val_y.shape[0],
            **self.model_kwargs
        )
        self.model.fit(
            series=self.train_data, covariates=self.cov_tr, **self.fit_kwargs
        )


###########################################################################################################################################################
# -------------------------------------Supporting external data methods------------------------------------------------------------------------------------
###########################################################################################################################################################


class SARIMAXTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_data(self):
        if self.exog_col is not None:
            self.train_data_exog = self.train_data[self.exog_col].copy()
            self.val_data_exog = self.val_data[self.exog_col].copy()
        else:
            self.train_data_exog = None
            self.val_data_exog = None

        self.train_data = self.train_data[self.col]
        self.val_data = self.val_data[self.col]

    def train(self):
        self.model = sm.tsa.statespace.SARIMAX(
            self.train_data, exog=self.train_data_exog, **self.model_kwargs
        )
        self.res = self.model.fit(**self.fit_kwargs)

    def predict(self):
        self.forecast = self.res.forecast(
            steps=self.val_data.shape[0], exog=self.val_data_exog
        )


class TCNTrainer(BaseDARTSTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.model = TCNModel(
            input_chunk_length=self.val_y.shape[0] * 3,
            output_chunk_length=self.val_y.shape[0],
            **self.model_kwargs
        )
        self.model.fit(
            series=self.train_data, covariates=self.cov_tr, **self.fit_kwargs
        )


class TransformerTrainer(BaseDARTSTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.model = TransformerModel(
            input_chunk_length=self.val_y.shape[0] * 3,
            output_chunk_length=self.val_y.shape[0],
            **self.model_kwargs
        )
        self.model.fit(
            series=self.train_data, covariates=self.cov_tr, **self.fit_kwargs
        )


###########################################################################################################################################################
# -------------------------------------TODO------------------------------------------------------------------------------------
###########################################################################################################################################################


# class GARCHTrainer(BaseTrainer):
#   def __init__(self, *args, **kwargs):
#       super().__init__(*args, **kwargs)

#   def preprocess_data(self):
#     self.train_data = self.train_data[self.col]
#     self.val_data = self.val_data[self.col]

#   def train(self):
#     self.model = arch_model(y=self.train_data, **self.model_kwargs)
#     self.res = self.model.fit(**self.fit_kwargs)

#   def predict(self):
#     self.forecast = self.res.forecast(horizon=self.val_data.shape[0])