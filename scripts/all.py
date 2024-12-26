import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from joblib import dump
from imblearn.over_sampling import SMOTE

# Step 1: Load the dataset
data_path = "data/data.csv"  # Replace with your dataset path
df = pd.read_csv(data_path)

# Step 2: Data Preprocessing
print("Dataset Preview:")
print(df.head())

# Drop unnecessary columns like 'id'
X = df.drop(columns=["diagnosis", "id"], axis=1)  # Features
y = df["diagnosis"]  # Target variable

# Encode 'diagnosis' column ('M' -> 1 for Malignant, 'B' -> 0 for Benign)
y = y.map({"M": 1, "B": 0})

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Step 3: Logistic Regression with Hyperparameter Tuning
logreg_param_grid = {
    "C": [0.1, 1, 10],  # Regularization strength
    "solver": ["liblinear", "newton-cg"],  # Solvers with better convergence
}
logreg = LogisticRegression(random_state=42, max_iter=5000)  # Increased max_iter
logreg_grid_search = GridSearchCV(logreg, logreg_param_grid, cv=5, scoring="accuracy")
logreg_grid_search.fit(X_train, y_train)
best_logreg_model = logreg_grid_search.best_estimator_
print("\nBest Logistic Regression Hyperparameters:", logreg_grid_search.best_params_)

# Predictions and evaluation for Logistic Regression
y_pred_logreg = best_logreg_model.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print("Logistic Regression ROC AUC:", roc_auc_score(y_test, y_pred_logreg))

# Step 4: Decision Tree Classifier with Hyperparameter Tuning
tree_param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "ccp_alpha": [0.01, 0.1, 0.5],  # Pruning parameter
}
tree = DecisionTreeClassifier(random_state=42)
tree_grid_search = GridSearchCV(tree, tree_param_grid, cv=5, scoring="accuracy")
tree_grid_search.fit(X_train, y_train)
best_tree_model = tree_grid_search.best_estimator_
print("\nBest Decision Tree Hyperparameters:", tree_grid_search.best_params_)

# Predictions and evaluation for Decision Tree
y_pred_tree = best_tree_model.predict(X_test)
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))
print("Decision Tree ROC AUC:", roc_auc_score(y_test, y_pred_tree))

# Cross-validation scores for both models
logreg_cv_scores = cross_val_score(
    best_logreg_model, X_resampled, y_resampled, cv=5, scoring="accuracy"
)
tree_cv_scores = cross_val_score(
    best_tree_model, X_resampled, y_resampled, cv=5, scoring="accuracy"
)
print("\nCross-Validation Accuracy Scores:")
print("Logistic Regression:", logreg_cv_scores)
print("Decision Tree:", tree_cv_scores)

# Step 5: Plot confusion matrices
ConfusionMatrixDisplay.from_estimator(best_logreg_model, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_estimator(best_tree_model, X_test, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Step 6: Plot ROC Curves


def plot_roc_curve(model, X_test, y_test, label):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")


plt.figure()
plot_roc_curve(best_logreg_model, X_test, y_test, "Logistic Regression")
plot_roc_curve(best_tree_model, X_test, y_test, "Decision Tree")
plt.show()

# Step 7: Feature Importance for Decision Tree
importances = best_tree_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance")
plt.show()

# Step 8: Save models
dump(best_logreg_model, "best_logreg_model.joblib")
dump(best_tree_model, "best_tree_model.joblib")

# Step 9: Exploratory Visualizations
# Distribution of 'diagnosis' variable
sns.countplot(data=df, x="diagnosis")
plt.title("Distribution of Diagnosis Variable")
plt.show()

# Scatter plot between 'radius_mean' and 'texture_mean'
sns.scatterplot(data=df, x="radius_mean", y="texture_mean", hue="diagnosis")
plt.title("Radius Mean vs Texture Mean")
plt.show()

# Pairplot for selected features
sns.pairplot(
    df,
    hue="diagnosis",
    vars=["radius_mean", "texture_mean", "perimeter_mean", "area_mean"],
)
plt.show()
