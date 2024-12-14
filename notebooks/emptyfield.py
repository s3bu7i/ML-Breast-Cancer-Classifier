import pandas as pd

data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Sütunların tiplərini yoxlayırıq
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Boş dəyərlərin təmizlənməsindən əvvəl vəziyyət
print("Missing values before filling:")
print(df.isnull().sum())

# Ədədi sütunlarda boş sahələri doldururuq (ortalama ilə)
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Kateqorial sütunlarda boş sahələri doldururuq (ən çox təkrarlanan dəyər ilə)
df[categorical_columns] = df[categorical_columns].fillna(
    df[categorical_columns].mode().iloc[0])

# Boş dəyərlərin təmizlənməsindən sonra vəziyyət
print("\nMissing values after filling:")
print(df.isnull().sum())

# Datasetin son versiyası haqqında məlumat
print("\nUpdated dataset info:")
print(df.info())

# Əgər kateqorial dəyişənlər maşın öyrənməsi üçün istifadə olunacaqsa:
# Kateqorial dəyişənləri one-hot kodlaşdırılması
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
