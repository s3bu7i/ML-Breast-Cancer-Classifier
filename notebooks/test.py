import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'data/data.csv'
data = pd.read_csv(file_path)

# Drop the 'id' column (since it's not a useful feature)
data = data.drop('id', axis=1)

# Convert the 'diagnosis' column to numeric using LabelEncoder (B = 0, M = 1)
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Split features and target
X = data.iloc[:, 1:]  # Features (all columns except 'diagnosis')
y = data['diagnosis']  # Target (the 'diagnosis' column)

# Split the dataset
