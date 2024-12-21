import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data_path = "data/data.csv"  # Path to the CSV dataset
df = pd.read_csv(data_path)  # Load the dataset into a pandas DataFrame

# Step 2: Preprocess the target variable
# Convert the 'diagnosis' column from categorical to numerical values
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Step 3: Prepare the feature matrix (X) and the target vector (y)
# Remove the target and identifier columns from the features
X = df.drop(columns=["diagnosis", "id"])
y = df["diagnosis"]

# Step 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the dataset into training and testing sets
# Use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 6: Initialize the models
# Logistic Regression with increased max_iter
logreg = LogisticRegression(random_state=42, max_iter=1000)
tree = DecisionTreeClassifier(random_state=42)  # Decision Tree model

# Step 7: Train the models on the training dataset
logreg.fit(X_train, y_train)  # Train the Logistic Regression model
tree.fit(X_train, y_train)  # Train the Decision Tree model

# Step 8: Make predictions on the test dataset
logreg_pred = logreg.predict(X_test)  # Predictions from Logistic Regression
tree_pred = tree.predict(X_test)  # Predictions from Decision Tree

# Step 9: Evaluate the models
# Calculate the accuracy of both models on the test dataset
logreg_acc = accuracy_score(y_test, logreg_pred)
tree_acc = accuracy_score(y_test, tree_pred)

# Print the accuracy of both models
print(f"Logistic Regression Accuracy: {logreg_acc:.4f}")
print(f"Decision Tree Accuracy: {tree_acc:.4f}")

# Generate and print classification reports
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logreg_pred))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, tree_pred))
