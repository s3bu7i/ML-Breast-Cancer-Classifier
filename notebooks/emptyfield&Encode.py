import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)

# Step 1: Load the dataset
data_path = "data/data.csv"  # Path to the dataset CSV file
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    logging.error(f"Error: File not found at {data_path}")
    exit()
except pd.errors.EmptyDataError:
    logging.error("Error: File is empty.")
    exit()

# Step 2: Identify numeric and categorical columns
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
categorical_columns = df.select_dtypes(include=["object"]).columns

# Step 3: Check for missing values before cleaning
logging.info("Missing values before filling:")
logging.info(df.isnull().sum())

# Step 4: Handle missing values in numeric columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 5: Handle missing values in categorical columns
df[categorical_columns] = df[categorical_columns].fillna(
    df[categorical_columns].mode().iloc[0])

# Step 6: Check for missing values after cleaning
logging.info("\nMissing values after filling:")
logging.info(df.isnull().sum())

# Step 7: Display updated dataset info
logging.info("\nUpdated dataset info:")
logging.info(df.info())

# Step 8: Scale numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Step 9: One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Step 10: Save processed data
output_path = "data/processed_data.csv"
df.to_csv(output_path, index=False)
logging.info(f"Processed dataset saved to {output_path}")
