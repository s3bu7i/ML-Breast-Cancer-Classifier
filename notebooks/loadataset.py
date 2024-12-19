import pandas as pd  # Importing the pandas library for data manipulation
import os  # Importing the os module to check file existence

# Step 1: Define the dataset file path
data_path = "data/data.csv"  # Specify the path to the dataset file

# Step 2: Check if the file exists
if not os.path.exists(data_path):  # Check if the file exists at the specified path
    print(f"Data file not found at: {data_path}")
else:
    try:
        # Step 3: Attempt to load the dataset
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(data_path)
        print("Dataset successfully loaded.")  # Confirm successful loading

        # Step 4: Preview the dataset
        print("Dataset preview:")  # Show the first few rows of the dataset
        print(data.head())  # Display the first 5 rows of the DataFrame

        # Step 5: Display dataset structure information
        # Provide structural information about the dataset
        print("\nDataset info:")
        # Output column names, types, non-null counts, and memory usage
        print(data.info())

    # Step 6: Handle potential errors
    except FileNotFoundError:  # Error when the file is not found
        print(f"File not found. Please check the file path: {data_path}")
    except pd.errors.EmptyDataError:  # Error when the file is empty
        print(f"The file is empty: {data_path}")
    except Exception as e:  # Catch any other unexpected errors
        print(f"An error occurred: {e}")
