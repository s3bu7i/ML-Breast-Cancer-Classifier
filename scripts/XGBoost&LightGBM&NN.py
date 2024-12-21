# Import necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import shap
from joblib import dump

# Load the dataset
data_path = "data/data.csv"  # Replace with your dataset path
data = pd.read_csv(data_path)

# Prepare features (X) and target (y)
X = data.drop(columns=["id", "diagnosis"])  # Drop 'id' as it's not useful
y = data["diagnosis"]

# Encode target labels ('B' -> 0, 'M' -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Apply feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# XGBoost Model with Hyperparameter Tuning
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
xgb = GridSearchCV(XGBClassifier(random_state=42),
                   xgb_param_grid, cv=5, scoring='accuracy', verbose=1)
xgb.fit(X_train, y_train)
best_xgb_model = xgb.best_estimator_
print("Best XGBoost Parameters:", xgb.best_params_)

y_pred_xgb = best_xgb_model.predict(X_test)
y_prob_xgb = best_xgb_model.predict_proba(X_test)[:, 1]
print("XGBoost Results:\n", classification_report(y_test, y_pred_xgb))

# LightGBM Model with Hyperparameter Tuning
lgbm_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [20, 31],
    'min_child_samples': [10, 20],
    'min_split_gain': [0.001, 0.01],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
lgbm = GridSearchCV(LGBMClassifier(random_state=42),
                    lgbm_param_grid, cv=5, scoring='accuracy', verbose=1)
lgbm.fit(X_train, y_train)
best_lgbm_model = lgbm.best_estimator_
print("Best LightGBM Parameters:", lgbm.best_params_)

y_pred_lgbm = best_lgbm_model.predict(X_test)
y_prob_lgbm = best_lgbm_model.predict_proba(X_test)[:, 1]
print("LightGBM Results:\n", classification_report(y_test, y_pred_lgbm))

# Neural Network Model with Improved Architecture
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,  # Increased regularization to avoid overfitting
    max_iter=2000,
    random_state=42
)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("Neural Network Results:\n", classification_report(y_test, y_pred_mlp))

# Cross-Validation for XGBoost
xgb_cv_scores = cross_val_score(best_xgb_model, X, y, cv=5, scoring='accuracy')
print("XGBoost Cross-Validation Accuracy:", xgb_cv_scores.mean())

# Confusion Matrix for XGBoost
ConfusionMatrixDisplay.from_estimator(best_xgb_model, X_test, y_test)
plt.title("Confusion Matrix - XGBoost")
plt.show()

# ROC Curve for XGBoost
fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)
plt.plot(fpr, tpr, label='XGBoost (AUC = {:.2f})'.format(
    roc_auc_score(y_test, y_prob_xgb)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.show()

# SHAP Explainability for XGBoost
explainer = shap.Explainer(best_xgb_model, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test)

# Save Models
dump(best_xgb_model, "best_xgb_model.joblib")
dump(best_lgbm_model, "best_lgbm_model.joblib")
dump(mlp, "best_mlp_model.joblib")
