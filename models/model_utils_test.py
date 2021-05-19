import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_utils import sarimax, garch

df = pd.read_csv("../data/raw_data.csv").rename(columns={"Hard coal consumption per capita [tones]": "consumption"})
n = df.shape[0]
df["exog1"] = df.consumption + np.random.rand(n)
df["exog2"] = np.random.rand(n)

split_idx = 294
df_train = df.iloc[:split_idx, :]
df_val = df.iloc[split_idx:, :]

print(df.head())

metric_garch, fcst_garch = garch(
    train=df_train,
    val=df_val,
    col="consumption",
    exog_col=["exog1", "exog2"]
)

