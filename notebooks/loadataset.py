import pandas as pd
import os

data_path = "data/data.csv"

if not os.path.exists(data_path):
    print(f"Data file not found at: {data_path}")
else:
    try:
        data = pd.read_csv(data_path)
        print("Dataset successfully loaded.")
        print("Dataset preview:")
        print(data.head())
        print("\nDataset info:")
        print(data.info())
    except FileNotFoundError:
        print(f"File not found. Please check the file path: {data_path}")
    except pd.errors.EmptyDataError:
        print(f"The file is empty: {data_path}")
    except Exception as e:
        print("An unexpected error occurred:", e)
