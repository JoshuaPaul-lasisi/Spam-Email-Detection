import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_models(X_train, y_train):
    """Train multiple models, save them to the models folder, and return a dictionary of trained models."""
    
    # Ensure the target folder exists
    model_folder = '../models'
    os.makedirs(model_folder, exist_ok=True)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(),
    }
    
    trained_models = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Save the model to the models folder
        model_path = os.path.join(model_folder, f'{name.replace(" ", "_").lower()}.pkl')
        joblib.dump(model, model_path)
        print(f'Model {name} saved to {model_path}')
    
    return trained_models

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance and return evaluation metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report