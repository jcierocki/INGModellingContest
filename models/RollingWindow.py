import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


class RollingWindow(object):
    def __init__(
        self, dataframe, data_frequency, mask, validation_start, forecast_horizon
    ) -> None:
        self.dataframe = dataframe
        self.mask = mask
        self.last_date = self.dataframe.index.max()
        self.data_frequency = data_frequency
        self.validation_start = validation_start
        self.forecast_horizon = forecast_horizon
        self.cursor = validation_start

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
        self.cursor = self.validation_start
        return self
