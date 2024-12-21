import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data_path = "data/data.csv"  # Update with your dataset path
df = pd.read_csv(data_path)

# Step 2: Prepare Features (X) and Target (y)
# Drop 'id' (irrelevant for prediction) and encode 'diagnosis' (target variable)
X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"]

# Encode the 'diagnosis' column ('M' -> 1 for Malignant, 'B' -> 0 for Benign)
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # 'B' -> 0, 'M' -> 1

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth=4, random_state=42)  # You can tune 'max_depth'
tree.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred_tree = tree.predict(X_test)

# Step 7: Evaluate the Model
# Classification Report
print("\nClassification Report for Decision Tree:")
print(classification_report(y_test, y_pred_tree))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_tree)
print("\nROC AUC Score:", roc_auc)

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tree)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Plot ROC Curve
if hasattr(tree, "predict_proba"):  # Check if predict_proba is available
    fpr, tpr, thresholds = roc_curve(y_test, tree.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label='Decision Tree (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Decision Tree')
    plt.legend()
    plt.show()

# Feature Importance Plot
feature_importances = tree.feature_importances_
features = df.drop(columns=["id", "diagnosis"]).columns
plt.barh(features, feature_importances)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.show()
