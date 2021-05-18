import sys

sys.path.append("..")

import fbprophet

from datetime import datetime
import pandas as pd
from hyperopt_utils import single_experiment
from model_utils import prophet, neural_prophet


def main():
    data_path = "data/raw_data.csv"
    dataframe = pd.read_csv(data_path)
    dataframe.index = dataframe["DATE"].astype("datetime64[ns]")

    out = single_experiment(
        dataframe,
        "month",
        datetime.strptime("2018-06-01", "%Y-%m-%d"),
        "Hard coal consumption per capita [tones]",
        6,
        "mse",
        neural_prophet,
    )

    print(out)


if __name__ == "__main__":
    main()
