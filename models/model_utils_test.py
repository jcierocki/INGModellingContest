import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_utils import sarimax, garch, prophet

HOR = 12

df = pd.read_csv("../data/raw_data.csv").rename(columns={"Hard coal consumption per capita [tones]": "consumption"})
n = df.shape[0]
df["exog1"] = df.consumption + np.random.rand(n)
df["exog2"] = np.random.rand(n)

df[["exog1", "exog2"]] = df[["exog1", "exog2"]].shift(HOR)
df = df.dropna()

df_train = df.iloc[:(n - 2 * HOR), :]
df_val = df.iloc[-HOR:, :]

# df_val = df.copy()
# df_val[["exog1", "exog2"]] = df_val[["exog1", "exog2"]].shift(HOR)
#
#
# print(df)
# print(df_val)

# metric_sarimax, fcst_sarimax = sarimax(
#     train=df_train,
#     val=df_val,
#     col="consumption",
#     exog_col=["exog1", "exog2"]
# )
#
# metric_garch, fcst_garch = garch(
#     train=df_train,
#     val=df_val,
#     col="consumption",
#     exog_col=["exog1", "exog2"]
# )

# metric_prophet, fcst_prophet = prophet(
#     train=df_train,
#     val=df_val,
#     col="consumption",
#     exog_col=["exog1", "exog2"]
# )

