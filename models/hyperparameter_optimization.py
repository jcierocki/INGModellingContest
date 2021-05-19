import sys

sys.path.append("..")

import fbprophet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hyperopt_utils import single_experiment_hyperopt
from model_utils import prophet, sarimax, var, ets
from hyperopt import fmin, tpe, space_eval, hp

from datetime import datetime


def main():
    data_path = "../data/raw_data.csv"
    dataframe = pd.read_csv(data_path)
    dataframe.index = dataframe["DATE"].astype("datetime64[ns]")

    space = {
        # "p": hp.choice("p", [0, 1, 2]),
        # "d": hp.choice("d", [0, 1, 2]),
        # "q": hp.choice("q", [0, 1, 2, 3]),
        "dataframe": hp.choice("dataframe", [dataframe]),
        "frequency": hp.choice("frequency", ["month"]),
        "train_end": hp.choice(
            "train_end", [datetime.strptime("2018-12-01", "%Y-%m-%d")]
        ),
        "forecast_column": hp.choice(
            "forecast_column", ["Hard coal consumption per capita [tones]"]
        ),
        "exog_columns": hp.choice("exog_columns", [None]),
        "forecast_freq": hp.choice("forecast_freq", ["MS"]),
        "forceast_horizon": hp.choice("forceast_horizon", [6]),
        "forecast_metric": hp.choice("forecast_metric", ["mae"]),
        "forecast_function": hp.choice("forecast_functions", [ets]),
    }

    best = fmin(single_experiment_hyperopt, space, algo=tpe.suggest, max_evals=5)
    best_vals = space_eval(space, best)
    print(best)
    print(best_vals)


if __name__ == "__main__":
    main()
