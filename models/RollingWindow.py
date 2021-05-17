import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


class RollingWindow(object):
    def __init__(
        self, dataframe, data_frequency, train_end, forecast_horizon
    ) -> None:
        self.dataframe = dataframe
        self.last_date = self.dataframe.index.max()
        self.data_frequency = data_frequency
        self.train_end = train_end
        self.forecast_horizon = forecast_horizon
        self.cursor = train_end

    def __next__(self):
        if self.data_frequency == "month":
            if (
                self.cursor + relativedelta(months=+self.forecast_horizon)
                > self.last_date
            ):
                raise StopIteration
            else:
                train_data = self.dataframe.query(f"index<='{str(self.cursor)}'").copy()
                test_data = self.dataframe.query(
                    f"index>'{str(self.cursor)}' and index<='{str(self.cursor + relativedelta(months=+self.forecast_horizon))}'"
                ).copy()
                self.cursor += relativedelta(months=+1)
                return train_data, test_data
        elif self.data_frequency == "week":
            if (
                self.cursor + relativedelta(weeks=+self.forecast_horizon)
                > self.last_date
            ):
                raise StopIteration
            else:
                train_data = self.dataframe.query(f"index<='{str(self.cursor)}'").copy()
                test_data = self.dataframe.query(
                    f"index>'{str(self.cursor)}' and index<='{str(self.cursor + relativedelta(weeks=+self.forecast_horizon))}'"
                ).copy()
                self.cursor += relativedelta(weeks=+1)
                return train_data, test_data

    def __iter__(self):
        self.cursor = self.train_end
        return self
