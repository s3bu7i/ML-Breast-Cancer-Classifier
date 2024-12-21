# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load your dataset from data/data.csv
data = pd.read_csv("data/data.csv")

# Define features (X) and target (y)
# Replace 'diagnosis' with your target column name if different
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

# Encode target labels ('B' -> 0, 'M' -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Print classification report
print("Random Forest Results:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Perform k-fold cross-validation
scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print("Cross-Validation Accuracy Scores:", scores)
print("Mean CV Accuracy:", scores.mean())
