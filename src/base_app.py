import streamlit as st
import pandas as pd
from data_preprocessor import load_data, preprocess_data
from feature_extraction import extract_features_and_target
from model_training import train_models, evaluate_model
from ensemble import ensemble_model
from visualization import visualize_results

st.title("Spam Email Classification")

# Load and preprocess the data
st.header("Load Dataset")
spam_df = load_data('../data/raw/spam.csv')
st.write(spam_df.head())

st.header("Preprocessed Data")
spam_df = preprocess_data(spam_df)
st.write(spam_df.head())

# Feature extraction
X, y = extract_features_and_target(spam_df)

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
st.header("Training Models")
trained_models = train_models(X_train, y_train)

# Evaluate models
st.header("Model Performance")
results = {}
for name, model in trained_models.items():
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
    results[name] = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

# Show results
visualize_results(results)

# Ensemble model
st.header("Ensemble Model")
ensemble_results = ensemble_model(X_train, y_train, X_test, y_test)
st.write("Ensemble Model Accuracy:", ensemble_results['accuracy'])
st.write("Ensemble Model Confusion Matrix:", ensemble_results['confusion_matrix'])
st.write("Ensemble Model Classification Report:", ensemble_results['classification_report'])