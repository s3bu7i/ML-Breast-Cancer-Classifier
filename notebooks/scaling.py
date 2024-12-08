# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# data_path = "data/data.csv"
# df = pd.read_csv(data_path)

# # Mövcud sütun adlarını yoxla
# print("Mövcud sütunlar:", df.columns)

# # 'target' adlı sütunun mövcudluğunu yoxla
# if "target" in df.columns:
#     X = df.drop(columns=["target"])
#     y = df["target"]
# else:
#     raise KeyError("'target' adlı sütun mövcud deyil. Dataset sütunlarını yoxlayın.")

# # Şkalaya salma
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# print("Şkalaya salınmış xüsusiyyətlər:\n", X_scaled)

import pandas as pd

# Load the uploaded dataset
file_path = 'data/data.csv'
df = pd.read_csv(file_path)

# Display the first few rows and summary information about the dataset
df_head = df.head()
df_info = df.info()
df_nulls = df.isnull().sum()

df_head, df_info, df_nulls
