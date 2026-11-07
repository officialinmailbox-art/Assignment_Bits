import os
import joblib

MODEL_DIR = "models"

model_files = {
    "Naive Bayes": "naive_bayes.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "SVM": "svm.pkl",
    "KNN": "knn.pkl",
    "Decision Tree": "decision_tree.pkl"
}

models = {}

for model_name, filename in model_files.items():
    model_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(model_path):
        models[model_name] = joblib.load(model_path)
