import pandas as pd
from models import *


df = pd.read_excel("../data/Lions_Den_data.xlsx")
df = pd.DataFrame(df)
df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "y"})
df = pd.DataFrame({"date": df["date"].values, "y": df["y"].values})
df.index = pd.to_datetime(df["date"])
df["exo"] = 1
df["exo1"] = 8


train, val = df.iloc[:300], df.iloc[300:310]

# exp = ETSTrainer(
#     train, val, col="y", model_kwargs={"initialization_method": "estimated"}
# )
# print(f"ETS {exp.run()[0]}")

# exp = ProphetTrainer(train, val, col="y")
# print(f"Prophet {exp.run()[0]}")

# exp = NeuralProphetTrainer(train, val, col="y")
# print(f"neural Prophet {exp.run()[0]}")

# exp = SARIMAXTrainer(train, val, col="y", exog_col=["exo", "exo1"])
# print(f"sarimax {exp.run()[0]}")

# exp = TBATSTrainer(train, val, col="y", exog_col=None)
# print(f"tbats {exp.run()[0]}")

# exp = TCNTrainer(train, val, col="y", exog_col=["exo", "exo1"])
# print(f"tcn {exp.run()[0]}")

exp = NBEATSTrainer(train, val, col="y", exog_col=None)
print(f"nbeats {exp.run()[0]}")

# exp = TransformerTrainer(train, val, col="y", exog_col=["exo", "exo1"])
# print(f"transformer {exp.run()[0]}")
