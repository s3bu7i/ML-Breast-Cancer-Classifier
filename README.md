
# Breast Cancer Classifier Using Machine Learning

This project is a Machine Learning-based breast cancer classifier that predicts whether a tumor is malignant or benign using key clinical data. The model is built with Python and leverages popular ML libraries.

---

## Features
- **Data Preprocessing**: Handles missing values, normalizes data, and prepares it for model training.
- **Model Training**: Uses supervised learning algorithms, including Logistic Regression, Support Vector Machines, and Random Forests.
- **Evaluation Metrics**: Includes accuracy, precision, recall, and F1 score for model performance evaluation.

---

## Dataset
The dataset used in this project is sourced from the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). It includes the following features:
- Mean radius, texture, perimeter, area, and more.
- Diagnosis: `M` (Malignant) or `B` (Benign).

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/s3bu7i/ML-Breast-Cancer-Classifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ML-Breast-Cancer-Classifier
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```
4. Predict new samples:
   ```bash
   python predict.py
   ```

---

## Model Performance
The classifier achieves high accuracy and reliability in distinguishing between malignant and benign cases. Below are the results of key evaluation metrics:
- **Accuracy**: 97%
- **Precision**: 96%
- **Recall**: 95%
- **F1 Score**: 95%

---

## Project Structure
```
ML-Breast-Cancer-Classifier/
├── data/                 # Dataset and preprocessing scripts
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for exploratory data analysis
├── scripts/              # Training and evaluation scripts
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Future Enhancements
- Implement deep learning models for improved performance.
- Explore feature selection and optimization techniques.
- Build a web application for real-time classification.

---

