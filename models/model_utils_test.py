import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_utils import sarimax, garch, prophet, ets, tbats, var

HOR = 12

df = pd.read_csv("../data/raw_data.csv").rename(columns={"Hard coal consumption per capita [tones]": "consumption"})
df.index = df["DATE"].astype("datetime64[ns]")
# df.index = pd.to_datetime(df["DATE"]).dt.date
n = df.shape[0]
df["exog1"] = df.consumption + np.random.rand(n)
df["exog2"] = np.random.rand(n)

df[["exog1", "exog2"]] = df[["exog1", "exog2"]].shift(HOR)
df = df.dropna()

df_train = df.iloc[:(n - 2 * HOR), :]
df_val = df.iloc[-HOR:, :]

# print(df)
# print(df_val)

metric_sarimax, fcst_sarimax = sarimax(
    train=df_train,
    val=df_val,
    col="consumption",
    exog_col=None
)

metric_sarimax, fcst_sarimax = sarimax(
    train=df_train,
    val=df_val,
    col="consumption",
    exog_col=["exog1", "exog2"]
)

metric_garch, fcst_garch = garch(
    train=df_train,
    val=df_val,
    col="consumption",
    exog_col=None
)

metric_garch, fcst_garch = garch(
    train=df_train,
    val=df_val,
    col="consumption",
    exog_col=["exog1", "exog2"]
)

metric_prophet, fcst_prophet = prophet(
    train=df_train,
    val=df_val,
    col="consumption",
    exog_col=None
)

metric_prophet, fcst_prophet = prophet(
    train=df_train,
    val=df_val,
    col="consumption",
    exog_col=["exog1", "exog2"]
)

metric_ets, fcst_ets = ets(
    train=df_train,
    val=df_val,
    col="consumption"
)

metric_tbats, fcst_tbats = tbats(
    train=df_train,
    val=df_val,
    col="consumption"
)

metric_var, fcst_var = var(
    train=df_train,
    val=df_val,
    col="consumption",
    exog_col=["exog1", "exog2"]
)
