import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data_path = "data/data.csv"  # Replace with your dataset path
df = pd.read_csv(data_path)

# Drop the 'id' column (it's not useful for predictions)
y = df["diagnosis"].map({"B": 0, "M": 1})
X = df.drop(columns=["diagnosis", "id"])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning function


def tune_hyperparameters(model, param_grid, x_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# Logistic Regression Hyperparameter Tuning
logreg_param_grid = {
    "C": [0.1, 1, 10],  # Regularization strength
    "solver": ["liblinear", "newton-cg"],  # Stable solvers
}
logreg = LogisticRegression(random_state=42, max_iter=5000)
best_logreg_model, best_logreg_params = tune_hyperparameters(
    logreg, logreg_param_grid, X_train, y_train
)
print("Best Logistic Regression Hyperparameters:", best_logreg_params)

# Decision Tree Hyperparameter Tuning
tree_param_grid = {
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "ccp_alpha": [0.01, 0.1, 0.5],
}
tree = DecisionTreeClassifier(random_state=42)
best_tree_model, best_tree_params = tune_hyperparameters(
    tree, tree_param_grid, X_train, y_train
)
print("Best Decision Tree Hyperparameters:", best_tree_params)

# Predictions and evaluation for both models
y_pred_logreg = best_logreg_model.predict(X_test)
y_pred_tree = best_tree_model.predict(X_test)

# Logistic Regression Evaluation
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Decision Tree Evaluation
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

# Plot Confusion Matrices
ConfusionMatrixDisplay.from_estimator(best_logreg_model, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_estimator(best_tree_model, X_test, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Function to Plot ROC Curve


def plot_roc_curve(model, x_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="blue", label=f"{model_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {model_name}")
        plt.legend(loc="lower right")
        plt.show()


# Plot ROC Curves
plot_roc_curve(best_logreg_model, X_test, y_test, "Logistic Regression")
plot_roc_curve(best_tree_model, X_test, y_test, "Decision Tree")
