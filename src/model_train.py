import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import joblib
import os

def train_and_save_models(data_path, models_dir):
    # Load preprocessed data
    df = pd.read_csv(data_path)
    
    # Extract features and target
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Message'])
    y = df['Spam']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(),
    }
    
    trained_models = {}
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        # Save the model
        model_path = os.path.join(models_dir, f'{name.replace(" ", "_").lower()}.pkl')
        joblib.dump(model, model_path)
        trained_models[name] = model
        print(f"{name} saved to {model_path}")
    
    # Ensemble model
    ensemble = VotingClassifier(estimators=[
        ('nb', MultinomialNB()),
        ('lr', LogisticRegression()),
        ('svm', SVC())
    ], voting='hard')
    ensemble.fit(X_train, y_train)
    
    # Save ensemble model
    ensemble_path = os.path.join(models_dir, 'ensemble_model.pkl')
    joblib.dump(ensemble, ensemble_path)
    print(f"Ensemble model saved to {ensemble_path}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    train_and_save_models('../data/processed/spam_data_processed.csv', '../models/')