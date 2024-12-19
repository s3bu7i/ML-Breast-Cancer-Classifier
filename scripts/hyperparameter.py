import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Drop the 'id' column (since it's not a useful feature)
y = df["diagnosis"].map({'B': 0, 'M': 1})
X = df.drop(columns=["diagnosis"])

# Data scaling (normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Logistic Regression
logreg_param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'solver': ['liblinear', 'saga']
}

tree_param_grid = {
    'max_depth': [3, 5, 10],  
    'min_samples_split': [2, 5, 10],  
    'ccp_alpha': [0.01, 0.1, 0.5]  
}

# Hyperparameter tuning function


def tune_hyperparameters(model, param_grid, x_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# logistic regression finding best model and hyperparameters
logreg = LogisticRegression(random_state=42, max_iter=1000)
best_logreg_model, best_logreg_params = tune_hyperparameters(
    logreg, logreg_param_grid, X_train, y_train)
print("Best Logistic Regression Hyperparameters:", best_logreg_params)

# Decision Tree finding best model and hyperparameters
tree = DecisionTreeClassifier(random_state=42)
best_tree_model, best_tree_params = tune_hyperparameters(
    tree, tree_param_grid, X_train, y_train)
print("Best Decision Tree Hyperparameters:", best_tree_params)

# Predictions and evaluation for both models
y_pred_logreg = best_logreg_model.predict(X_test)
y_pred_tree = best_tree_model.predict(X_test)

# Classification Reports
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

# ROC AUC Scores
ConfusionMatrixDisplay.from_estimator(best_logreg_model, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_estimator(best_tree_model, X_test, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# ROC Curve and AUC


def plot_roc_curve(model, x_test, y_test, model_name):
    fpr, tpr, _ = roc_curve(
        y_test, model.predict_proba(x_test)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue',
             label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()


# Plot ROC curves for both models
plot_roc_curve(best_logreg_model, X_test, y_test, "Logistic Regression")
plot_roc_curve(best_tree_model, X_test, y_test, "Decision Tree")
