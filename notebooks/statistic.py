import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Load dataset
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Drop 'id' column as it's not useful for prediction
df = df.drop(columns=["id"])

# Map 'diagnosis' to numeric values (e.g., 0 for benign, 1 for malignant)
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Split data into features and target
X = df.drop(columns=["diagnosis"])  # Features
y = df["diagnosis"]  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the models
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train_scaled, y_train)

# ROC Curve for Logistic Regression
fpr, tpr, _ = roc_curve(y_test, logreg.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr, tpr, label="Logistic Regression")

# ROC Curve for Decision Tree
fpr_tree, tpr_tree, _ = roc_curve(
    y_test, tree.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr_tree, tpr_tree, label="Decision Tree")

# Adding labels and title
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

# Adding legend to distinguish between the models
plt.legend()

# Display the plot
plt.show()
