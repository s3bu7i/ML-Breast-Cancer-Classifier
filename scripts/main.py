import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Dataseti yükləyirik
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Hədəf və xüsusiyyətləri ayırırıq
X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"]

# Kateqorial hədəf dəyişənini şifrələyirik
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # M=1, B=0

# Məlumatı train/test olaraq bölürük
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model qururuq
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# Təxminlər
y_pred_tree = tree.predict(X_test)

# Nəticələri göstəririk
print(classification_report(y_test, y_pred_tree))
print("ROC AUC:", roc_auc_score(y_test, y_pred_tree))