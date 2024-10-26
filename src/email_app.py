import streamlit as st
import pandas as pd
from model_training import train_models, evaluate_model
from ensemble import ensemble_model
from feature_extraction import extract_features_and_target
from data_preprocessor import preprocess_data
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Custom Spam Email Classifier")

# Step 1: Accept input as email link or text
st.header("Input Email")

# Create radio buttons to choose the input method
input_option = st.radio("Choose how to input your email:", ('Email Link', 'Text Input'))

email_text = ""

if input_option == 'Email Link':
    # User provides an email link
    email_link = st.text_input("Enter the email link:")
    if st.button("Fetch Email"):
        # Simulate fetching email from the link (this would require integration with an email service API)
        email_text = "This is a simulated email fetched from the link."  # Replace with actual fetching logic
        st.write(f"Fetched Email Content: {email_text}")
else:
    # User enters the email text manually
    email_text = st.text_area("Paste the email content here:")

if email_text:
    # Step 2: Set user preferences for non-spam mail
    st.header("Set Preferences for Non-Spam Emails")
    
    # Example preferences
    keywords = st.text_area("Enter keywords that suggest an email is not spam (comma-separated):", value="invoice, meeting, thank you")
    trusted_senders = st.text_area("Enter trusted senders (comma-separated):", value="example@trusted.com, client@domain.com")
    
    # Process preferences
    keyword_list = [word.strip().lower() for word in keywords.split(',')]
    trusted_senders_list = [sender.strip().lower() for sender in trusted_senders.split(',')]

    # Step 3: Preprocess the email content
    st.header("Preprocessed Email")
    email_df = pd.DataFrame({'Message': [email_text], 'Category': ['unknown']})  # Dummy category, for processing purposes
    preprocessed_email_df = preprocess_data(email_df)
    st.write(preprocessed_email_df)

    # Feature extraction (we assume feature extraction is applied based on preprocessed data)
    X, _ = extract_features_and_target(preprocessed_email_df)

    # Step 4: Load a pre-trained model (assuming it's trained elsewhere) or train a model with sample data
    # Simulated pre-trained model loading (replace with actual loading logic)
    X_train, X_test, y_train, y_test = train_test_split(X, [0], test_size=0.3, random_state=42)  # Dummy target for illustration
    trained_models = train_models(X_train, y_train)  # Use actual training if needed

    # Step 5: Run the model on the new email
    st.header("Model Prediction")
    for name, model in trained_models.items():
        accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
        st.write(f"Model: {name}")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Confusion Matrix:\n{conf_matrix}")
        st.write(f"Classification Report:\n{class_report}")
    
    # Step 6: Ensemble model prediction
    ensemble_results = ensemble_model(X_train, y_train, X_test, y_test)
    st.write("Ensemble Model Results")
    st.write("Ensemble Model Accuracy:", ensemble_results['accuracy'])
    st.write("Ensemble Model Confusion Matrix:", ensemble_results['confusion_matrix'])
    st.write("Ensemble Model Classification Report:", ensemble_results['classification_report'])
else:
    st.info("Please input the email content or a link to proceed.")