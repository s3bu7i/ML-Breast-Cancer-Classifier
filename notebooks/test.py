from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

data_path = "data/data.csv"
df = pd.read_csv(data_path)

df = df.drop(columns=["id"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
