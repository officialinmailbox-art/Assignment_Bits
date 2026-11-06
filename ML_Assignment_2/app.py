
import streamlit as st

import pandas as pd

st.set_page_config(
    page_title="ML Assignment 2",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classification")
st.subheader("Adult Census Income Dataset")

st.write(
    "This application demonstrates classification "
    "models for predicting income."
)

st.info("Welcome to ML Assignment 2!")

st.sidebar.header("Project Information")

st.sidebar.write("Models:")
st.sidebar.write("- Logistic Regression")
st.sidebar.write("- Decision Tree")
st.sidebar.write("- KNN")
st.sidebar.write("- Naive Bayes")
st.sidebar.write("- SVM")

st.success("Streamlit application is running successfully!")

