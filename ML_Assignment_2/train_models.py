
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Load training dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(BASE_DIR, "adult_train.csv")
)

# Clean column names and values
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df = df.replace("?", pd.NA).dropna()

# Convert target
df["income"] = (
    df["income"].astype(str).str.rstrip(".") == ">50K"
).astype(int)

X = df.drop(columns=["income"])
y = df["income"]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), cat_cols)
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", C=1.0)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs("models", exist_ok=True)

results = []

for name, classifier in models.items():

    pipeline = Pipeline([
        ("pre", make_preprocessor()),
        ("clf", classifier)
    ])

    # Train only on training data
    pipeline.fit(X_train, y_train)

    # Save complete preprocessing + model pipeline
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(pipeline, os.path.join("models", filename))

    predictions = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "MCC": matthews_corrcoef(y_test, predictions)
    })

    print("Saved:", name)

results_df = pd.DataFrame(results).set_index("Model").round(4)
results_df.to_csv("model_results.csv")

print("\nModel comparison:")
print(results_df)

print("\nAll five models saved successfully!")
