import pandas as pd

data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Kateqorial və ədədi sütunları ayırırıq
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Yalnız ədədi sütunlar üçün boş sahələri doldururuq
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Kateqorial sütunlarda boş dəyərlər varsa, onları da doldurmaq lazımdır
df[categorical_columns] = df[categorical_columns].fillna(
    df[categorical_columns].mode().iloc[0])

# Datasetin yenilənmiş versiyasını yoxlayın
print(df.info())
print(df.isnull().sum())
