import streamlit as st
import joblib

# Load the models and vectorizer
@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("models/vectorizer.pkl")

# Title
st.title("Spam Email Classifier")

# Text input for new email
email_input = st.text_area("Enter the email content:")

# Select the model
model_choice = st.selectbox("Choose the model:", ["Logistic Regression", "Naive Bayes", "SVM", "Ensemble Model"])

# Predict button
if st.button("Classify"):
    model_mapping = {
        "Logistic Regression": "logistic_regression",
        "Naive Bayes": "naive_bayes",
        "SVM": "svm",
        "Ensemble Model": "ensemble_model"
    }
    model = load_model(model_mapping[model_choice])
    vectorizer = load_vectorizer()
    
    # Transform and predict
    email_transformed = vectorizer.transform([email_input])
    prediction = model.predict(email_transformed)[0]
    
    # Display the result
    result = "Spam" if prediction == 1 else "Not Spam"
    st.write(f"The email is classified as: {result}")