# Import necessary libraries
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Load the dataset
# Replace 'data/data.csv' with the correct path to your dataset
data = pd.read_csv("data/data.csv")

# Define features (X) and target (y)
# Replace 'diagnosis' with the actual target column name if different
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

# Encode target labels ('B' -> 0, 'M' -> 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create pipeline
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),  # Scale the features
        # Random Forest Classifier
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Fit and evaluate the pipeline
pipeline.fit(X_train, y_train)
y_pred_pipeline = pipeline.predict(X_test)

# Print the classification report
print("Pipeline Results:\n", classification_report(y_test, y_pred_pipeline))
