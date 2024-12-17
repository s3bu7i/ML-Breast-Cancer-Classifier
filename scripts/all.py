import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc

# Load the dataset
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Check the first few rows of the dataset to understand the columns
print(df.head())

# Ensure that the 'diagnosis' column is the target variable
X = df.drop(columns=["diagnosis"])  # Drop diagnosis column for features
y = df["diagnosis"]  # Use diagnosis as the target variable

# Encode the 'diagnosis' column if it's categorical (e.g., 'M' for malignant, 'B' for benign)
# Assuming 'M' = malignant and 'B' = benign, adjust if needed
y = y.map({"M": 1, "B": 0})

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Logistic Regression
logreg_param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'solver': ['liblinear', 'saga']
}
logreg = LogisticRegression(random_state=42)
logreg_grid_search = GridSearchCV(
    logreg, logreg_param_grid, cv=5, scoring='accuracy')
logreg_grid_search.fit(X_train, y_train)

# Best Logistic Regression model
best_logreg_model = logreg_grid_search.best_estimator_
print("Best Logistic Regression Hyperparameters:",
      logreg_grid_search.best_params_)

# Predictions and evaluation for Logistic Regression
y_pred_logreg = best_logreg_model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print("Logistic Regression ROC AUC:", roc_auc_score(y_test, y_pred_logreg))

# Hyperparameter tuning for Decision Tree Classifier
tree_param_grid = {
    'max_depth': [3, 5, 10, None],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Min samples for a split
    'ccp_alpha': [0.01, 0.1, 0.5]  # Pruning parameter
}
tree = DecisionTreeClassifier(random_state=42)
tree_grid_search = GridSearchCV(
    tree, tree_param_grid, cv=5, scoring='accuracy')
tree_grid_search.fit(X_train, y_train)

# Best Decision Tree model
best_tree_model = tree_grid_search.best_estimator_
print("Best Decision Tree Hyperparameters:", tree_grid_search.best_params_)

# Predictions and evaluation for Decision Tree
y_pred_tree = best_tree_model.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))
print("Decision Tree ROC AUC:", roc_auc_score(y_test, y_pred_tree))

# Cross-validation scores for both models
logreg_cv_scores = cross_val_score(
    best_logreg_model, X_scaled, y, cv=5, scoring='accuracy')
tree_cv_scores = cross_val_score(
    best_tree_model, X_scaled, y, cv=5, scoring='accuracy')

print("Logistic Regression Cross-Validation Scores:", logreg_cv_scores)
print("Decision Tree Cross-Validation Scores:", tree_cv_scores)

# Plotting confusion matrices for both models
# Logistic Regression Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best_logreg_model, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Decision Tree Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best_tree_model, X_test, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# ROC Curve visualization for both models


def plot_roc_curve(model, X_test, y_test, label):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {label}')
    plt.legend(loc='lower right')
    plt.show()


# Plot ROC Curve for Logistic Regression
plot_roc_curve(best_logreg_model, X_test, y_test, "Logistic Regression")

# Plot ROC Curve for Decision Tree
plot_roc_curve(best_tree_model, X_test, y_test, "Decision Tree")

# Visualize distribution of target variable ('diagnosis')
sns.countplot(data=df, x="diagnosis")
plt.title("Distribution of Diagnosis Variable")
plt.show()

# Visualize the relationship between 'radius_mean' and 'texture_mean'
sns.scatterplot(data=df, x="radius_mean", y="texture_mean", hue="diagnosis")
plt.title("Radius Mean vs Texture Mean Distribution")
plt.show()


