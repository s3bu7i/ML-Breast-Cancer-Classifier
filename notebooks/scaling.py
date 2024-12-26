import pandas as pd

# Load the uploaded dataset
file_path = "data/data.csv"
df = pd.read_csv(file_path)

# Display the first few rows and summary information about the dataset
df_head = df.head()
df_info = df.info()
df_nulls = df.isnull().sum()

print(df_head)
print(df_info)
print(df_nulls)
