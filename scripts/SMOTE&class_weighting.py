# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the dataset
# Replace 'data/data.csv' with your dataset path
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

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest with class weighting
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
rf.fit(X_resampled, y_resampled)
y_pred_rf = rf.predict(X_test)

# Print classification report
print("Random Forest with SMOTE Results:\n",
      classification_report(y_test, y_pred_rf))

# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Confusion Matrix - Random Forest with SMOTE")
plt.show()
