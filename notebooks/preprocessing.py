import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Load the dataset
data_path = "data/data.csv"  # Path to your dataset
data = pd.read_csv(data_path)

# Step 4: Encode categorical variables (e.g., 'diagnosis')
# 'M' -> 1, 'B' -> 0
label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Step 5: Handle missing values
# Impute missing values for numeric columns with the median
data.fillna(data.median(), inplace=True)

# Step 6: Split dataset into features (X) and target (y)
# Drop 'diagnosis' and 'id' columns from features
X = data.drop(columns=["diagnosis", "id"], axis=1)
y = data["diagnosis"]  # Target column

# Step 7: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Display processed data
print("\nProcessed Features (X):")
print(X_scaled[:5])  # Display first 5 rows of scaled features
print("\nProcessed Target (y):")
print(y.head())  # Display first 5 rows of the target variable
