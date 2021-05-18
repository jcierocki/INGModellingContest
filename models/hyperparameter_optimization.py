import sys

# sys.path.append("..")

import fbprophet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datetime import datetime
from hyperopt import fmin, tpe, space_eval, hp
from hyperopt.pyll.base import scope

from hyperopt_utils import single_experiment
from model_utils import prophet, neural_prophet
from eda_utils import stl_ets, optimize_stl


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


if __name__ == "__main__":
    main()
