import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Dataseti yükləyirik
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# Hədəf və xüsusiyyətləri ayırırıq
X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"]

# Kateqorial hədəf dəyişənini şifrələyirik
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Məlumatı train/test olaraq bölürük
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model qururuq
tree = DecisionTreeClassifier(max_depth=4, random_state=42, ccp_alpha=0.01)
tree.fit(X_train, y_train)

# Təxminlər
y_pred_tree = tree.predict(X_test)

# Nəticələri göstəririk
print(classification_report(y_test, y_pred_tree))
print("ROC AUC:", roc_auc_score(y_test, y_pred_tree))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tree)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
            "Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.title("Decision Tree Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, tree.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='Decision Tree')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
