from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load your dataset
data = pd.read_csv("data/data.csv")  # Update the path
X = data.drop(columns=["diagnosis"])  # Replace "target" with your label column
y = data["diagnosis"]

# Recursive Feature Elimination
model = RandomForestClassifier(random_state=42)
# Adjust features to select
rfe = RFE(estimator=model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
print("Selected Features with RFE:", X.columns[rfe.support_])

# Mutual Information
mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)
print("Mutual Information Scores:\n", mi_scores.sort_values(ascending=False))
