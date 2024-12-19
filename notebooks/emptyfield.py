import pandas as pd

# Step 1: Load the dataset
data_path = "data/data.csv"  # Path to the dataset CSV file
df = pd.read_csv(data_path)  # Load the dataset into a pandas DataFrame

# Step 2: Identify numeric and categorical columns
# Select numeric columns (float64 and int64 types)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Select categorical columns (object type, typically strings)
categorical_columns = df.select_dtypes(include=['object']).columns

# Step 3: Check for missing values before cleaning
print("Missing values before filling:")
print(df.isnull().sum())  # Count of missing values in each column

# Step 4: Handle missing values in numeric columns
# Fill missing values in numeric columns with the column's mean
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 5: Handle missing values in categorical columns
# Fill missing values in categorical columns with the most frequent value (mode)
df[categorical_columns] = df[categorical_columns].fillna(
    df[categorical_columns].mode().iloc[0])  # `.iloc[0]` gets the top mode value

# Step 6: Check for missing values after cleaning
print("\nMissing values after filling:")
print(df.isnull().sum())  # Verify there are no missing values left

# Step 7: Display updated dataset info
print("\nUpdated dataset info:")
# Summary of the DataFrame, including column types and non-null counts
print(df.info())

# Step 8: One-hot encode categorical columns (if needed for machine learning)
# Convert categorical variables into dummy/indicator variables
# `drop_first=True` avoids multicollinearity by dropping the first dummy column for each category
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
