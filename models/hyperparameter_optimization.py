import sys

# sys.path.append("..")

import fbprophet

from datetime import datetime
import pandas as pd
import numpy as np
from hyperopt_utils import single_experiment
from model_utils import prophet, neural_prophet
from stl_utils import stl_ets
from hyperopt import fmin, tpe, space_eval, hp
from hyperopt.pyll.base import scope


def main():
    data_path = "../data/raw_data.csv"
    dataframe = pd.read_csv(data_path)
    dataframe.index = dataframe["DATE"].astype("datetime64[ns]")
    dataframe = dataframe.iloc[0:306, :]

    out = single_experiment(
        dataframe,
        "month",
        datetime.strptime("2017-12-01", "%Y-%m-%d"),
        "Hard coal consumption per capita [tones]",
        24,
        "mae",
        stl_ets
    )

    print(out)

    def objective(args):
        return single_experiment(
            dataframe,
            "month",
            datetime.strptime("2017-12-01", "%Y-%m-%d"),
            "Hard coal consumption per capita [tones]",
            24,
            "mae",
            stl_ets,
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

    best = fmin(objective, space, algo=tpe.suggest, max_evals=200)

    print(best)


if __name__ == "__main__":
    main()
