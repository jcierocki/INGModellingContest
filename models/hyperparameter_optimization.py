import sys

sys.path.append("..")

import fbprophet

from datetime import datetime
import pandas as pd
import numpy as np
from hyperopt_utils import single_experiment_hyperopt
from model_utils import prophet, neural_prophet, sarima
from hyperopt import fmin, tpe, space_eval, hp


def main():
    data_path = "../data/raw_data.csv"
    dataframe = pd.read_csv(data_path)
    dataframe.index = dataframe["DATE"].astype("datetime64[ns]")
    dataframe = dataframe.iloc[0:306, :]

    space = {
        "p": hp.choice("p", [0, 1, 2]),
        "d": hp.choice("d", [0, 1, 2]),
        "q": hp.choice("q", [0, 1, 2, 3]),
        "s": hp.choice("s", [2, 3]),
        "dataframe": hp.choice("dataframe", [dataframe]),
        "frequency": hp.choice("frequency", ["month"]),
        "train_end": hp.choice(
            "train_end", [datetime.strptime("2018-12-01", "%Y-%m-%d")]
        ),
        "forecast_column": hp.choice(
            "forecast_column", ["Hard coal consumption per capita [tones]"]
        ),
        "forceast_horizon": hp.choice("forceast_horizon", [6]),
        "forecast_metric": hp.choice("forecast_metric", ["mae"]),
        "forecast_function": hp.choice("forecast_functions", [sarima]),
    }

    best = fmin(single_experiment_hyperopt, space, algo=tpe.suggest, max_evals=5)
    best_vals = space_eval(space, best)
    print(best)
    print(best_vals)


if __name__ == "__main__":
    main()
