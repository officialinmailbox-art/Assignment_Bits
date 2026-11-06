
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Classification")
st.subheader("Adult Census Income Prediction")

st.write(
    "Evaluate five trained machine learning models "
    "using your uploaded test dataset."
)

st.info(
    "Models are loaded from saved files. "
    "No model training takes place in this application."
)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_DIR = "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded = {}

    for model_name, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(path):
            loaded[model_name] = joblib.load(path)

    return loaded


try:
    models = load_models()
except Exception as error:
    st.error(f"Unable to load models: {error}")
    st.stop()

if len(models) != 5:

    missing = set(MODEL_FILES) - set(models)

    st.error(
        "Some saved models are missing: "
        + ", ".join(missing)
    )

    st.info("Please run train_models.py and upload all five model files.")
    st.stop()

st.success("All five saved models loaded successfully.")

# --------------------------------------------------
# CSV UPLOAD
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose your Adult Census test CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.warning("Please upload your test CSV dataset to continue.")
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read CSV: {error}")
    st.stop()

test_df.columns = test_df.columns.str.strip()

for col in test_df.select_dtypes(include="object").columns:
    test_df[col] = test_df[col].str.strip()

st.write("Uploaded dataset shape:", test_df.shape)

with st.expander("Preview uploaded dataset"):
    st.dataframe(test_df.head(10), use_container_width=True)

# --------------------------------------------------
# CLEAN TEST DATA
# --------------------------------------------------

if "income" not in test_df.columns:
    st.error("Target column 'income' is missing from your CSV.")
    st.stop()

test_df = test_df.replace("?", pd.NA).dropna().copy()

# Support both original Adult labels and numeric labels
income_values = test_df["income"].astype(str).str.strip().str.rstrip(".")

label_mapping = {
    "<=50K": 0,
    ">50K": 1,
    "0": 0,
    "1": 1
}

unknown_labels = set(income_values.unique()) - set(label_mapping)

if unknown_labels:
    st.error(f"Unknown income labels: {unknown_labels}")
    st.stop()

y_test = income_values.map(label_mapping).astype(int)

X_test = test_df.drop(columns=["income"])

# --------------------------------------------------
# VALIDATE INPUT FEATURES
# --------------------------------------------------

first_model = next(iter(models.values()))

expected_columns = list(first_model.feature_names_in_)

missing_columns = set(expected_columns) - set(X_test.columns)

if missing_columns:
    st.error(
        "Missing required input features: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

# Match training feature order and ignore unrelated extra columns
X_test = X_test[expected_columns]

if len(X_test) == 0:
    st.error("No valid rows remain after removing missing values.")
    st.stop()

st.success(f"Test dataset ready: {len(X_test)} records.")

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

st.header("2. Select Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a model",
    list(MODEL_FILES.keys())
)

selected_model = models[selected_model_name]

# --------------------------------------------------
# EVALUATE ALL FIVE MODELS
# --------------------------------------------------

st.header("3. Model Evaluation")

all_results = []
predictions_by_model = {}
scores_by_model = {}

try:

    for model_name, model in models.items():

        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)

        predictions_by_model[model_name] = predictions
        scores_by_model[model_name] = scores

        all_results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, scores),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, predictions)
        })

except Exception as error:
    st.error(f"Evaluation failed: {error}")
    st.stop()

results_df = pd.DataFrame(all_results).set_index("Model")

# --------------------------------------------------
# SELECTED MODEL METRICS
# --------------------------------------------------

st.subheader(f"Performance: {selected_model_name}")

selected_result = results_df.loc[selected_model_name]

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", f"{selected_result['Accuracy']:.4f}")
c2.metric("Precision", f"{selected_result['Precision']:.4f}")
c3.metric("Recall", f"{selected_result['Recall']:.4f}")

c4, c5, c6 = st.columns(3)

c4.metric("F1 Score", f"{selected_result['F1']:.4f}")
c5.metric("AUC", f"{selected_result['AUC']:.4f}")
c6.metric("MCC", f"{selected_result['MCC']:.4f}")

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.header("4. Comparison of All Five Models")

st.dataframe(
    results_df.round(4),
    use_container_width=True
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.header("5. Confusion Matrix")

selected_predictions = predictions_by_model[selected_model_name]

cm = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["<=50K", ">50K"],
    yticklabels=["<=50K", ">50K"],
    ax=ax
)

ax.set_xlabel("Predicted Income")
ax.set_ylabel("Actual Income")
ax.set_title(selected_model_name)

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.header("6. Classification Report")

report = classification_report(
    y_test,
    selected_predictions,
    labels=[0, 1],
    target_names=["<=50K", ">50K"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.header("7. Download Evaluation Results")

csv_data = results_df.to_csv().encode("utf-8")

st.download_button(
    label="Download Model Comparison CSV",
    data=csv_data,
    file_name="model_comparison_results.csv",
    mime="text/csv"
)

st.success("Evaluation completed successfully!")
