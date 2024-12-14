import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np


data_path = "data/data.csv"
data = pd.read_csv(data_path)
# Display basic info and statistics
import numpy as np

# Display basic info and statistics
data.info()
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Handle missing values (example: impute with median)
data.fillna(data.median(), inplace=True)

# Encode categorical variables if needed
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Split dataset into features and target
# Update 'target_column' with the actual column name
X = data.drop(columns=["diagnosis"], axis=1)
y = data["diagnosis"]

# Standardize features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
