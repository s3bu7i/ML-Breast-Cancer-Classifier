import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
data_path = "data/data.csv"
df = pd.read_csv(data_path)
# Dataseti təkrar yükləyirik və əvvəlki addımları yerinə yetiririk

# 'diagnosis' sütununu numerik dəyərlərə çevirmək
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# X və y dəyişənlərini yaratmaq
X = df.drop(columns=["diagnosis", "id"])  # Features
y = df["diagnosis"]  # Target variable

# Dataseti tren və test hissələrinə bölmək
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model Seçimi: Logistic Regression və Decision Tree
logreg = LogisticRegression(random_state=42)
tree = DecisionTreeClassifier(random_state=42)

logreg.fit(X_train, y_train)
tree.fit(X_train, y_train)

# Test dəstində qiymətləndirmə
logreg_pred = logreg.predict(X_test)
tree_pred = tree.predict(X_test)

# Hər iki modelin nəticələrinin müqayisəsi
logreg_acc = accuracy_score(y_test, logreg_pred)
tree_acc = accuracy_score(y_test, tree_pred)

print(f"Logistic Regression Accuracy: {logreg_acc:.4f}")
print(f"Decision Tree Accuracy: {tree_acc:.4f}")

# Hər iki modelin klassifikasiya hesabatı
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logreg_pred))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, tree_pred))
