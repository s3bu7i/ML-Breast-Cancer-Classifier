import pandas as pd

data_path = "data/data.csv"
df = pd.read_csv(data_path)

print(df.head())
