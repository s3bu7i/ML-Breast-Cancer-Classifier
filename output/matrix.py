import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay

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

# Logistic Regression Model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# Predictions and evaluation for Logistic Regression
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print("Logistic Regression ROC AUC:", roc_auc_score(y_test, y_pred_logreg))

# Decision Tree Classifier Model
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# Predictions and evaluation for Decision Tree
y_pred_tree = tree.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))
print("Decision Tree ROC AUC:", roc_auc_score(y_test, y_pred_tree))

# Plotting confusion matrices for both models
# Logistic Regression Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Decision Tree Confusion Matrix
ConfusionMatrixDisplay.from_estimator(tree, X_test, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Visualize distribution of target variable ('diagnosis')
sns.countplot(data=df, x="diagnosis")
plt.title("Distribution of Diagnosis Variable")
plt.show()

# Visualize the relationship between 'radius_mean' and 'texture_mean'
sns.scatterplot(data=df, x="radius_mean", y="texture_mean", hue="diagnosis")
plt.title("Radius Mean vs Texture Mean Distribution")
plt.show()
