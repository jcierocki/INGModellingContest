import sys

sys.path.append("..")

import pandas as pd
import multiprocessing as mp
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pytz import UTC

from RollingWindow import RollingWindow


def example_function(train, test):
    return (
        train.shape,
        test.shape,
        train.index.max(),
        test.index.min(),
        test.index.max(),
    )


def main():
    data_path = "data/raw_data.csv"
    df = pd.read_csv(data_path)
    df.index = pd.to_datetime(df["DATE"])
    rw = RollingWindow(
        df,
        "month",
        #"%Y-%m-%dT%H:%M:%SZ",
        datetime.strptime("2018-06-01", "%Y-%m-%d").replace(tzinfo=UTC),
        3,
    )
    with mp.Pool(mp.cpu_count() - 2) as p:
        result = p.starmap_async(
            example_function, ((train, test) for train, test in rw)
        )
        print(result.get())


if __name__ == "__main__":
    main()
