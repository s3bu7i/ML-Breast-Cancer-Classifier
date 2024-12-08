import pandas as pd

# Dataset yüklənməsi
data = pd.read_csv("data.csv")

missing_values = data.isnull().sum()
missing_values.to_csv("./analyse_data/missing_values.csv")
stats = data.describe()
stats.to_csv("./analyse_data/output.csv", index=True)
