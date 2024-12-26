import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "data/data.csv"
data = pd.read_csv(file_path)

# Drop 'id' if it exists as it's not a useful feature
if "id" in data.columns:
    data = data.drop("id", axis=1)

# Convert 'diagnosis' to numeric using LabelEncoder (B = 0, M = 1)
if "diagnosis" in data.columns:
    label_encoder = LabelEncoder()
    data["diagnosis"] = label_encoder.fit_transform(data["diagnosis"])

# Assuming 'diagnosis' is the target, ensure 'X' and 'y' are selected correctly
X = data.drop("diagnosis", axis=1)  # Features
y = data["diagnosis"]  # Target (should now be categorical 0, 1)

# Print unique values of 'y' to ensure it contains 0s and 1s
print(f"Unique values in target y: {y.unique()}")

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the DecisionTreeClassifier and set up the hyperparameter grid
decision_tree = DecisionTreeClassifier(random_state=42)
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    estimator=decision_tree, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best Parameters from Grid Search:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Train the DecisionTreeClassifier with the best parameters
best_decision_tree = grid_search.best_estimator_
best_decision_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_decision_tree.predict(X_test)

# Calculate the confusion matrix and accuracy score
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print the confusion matrix and accuracy
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)

# Optional: Print a detailed classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
