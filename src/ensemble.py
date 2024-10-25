import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from feature_extraction import extract_features_and_target
from data_preprocessor import load_data

def ensemble_model(X_train, y_train, X_test, y_test):
    """Train and evaluate an ensemble model using VotingClassifier."""
    ensemble = VotingClassifier(estimators=[
        ('nb', MultinomialNB()),
        ('lr', LogisticRegression()),
        ('svm', SVC())
    ], voting='hard')
    
    start_time = time.time()
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    end_time = time.time()
    
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    conf_matrix = confusion_matrix(y_test, y_pred_ensemble)
    class_report = classification_report(y_test, y_pred_ensemble)
    runtime = end_time - start_time
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'runtime': runtime
    }

def save_evaluation_results(results, filepath):
    """Save model evaluation results to a text file."""
    # Ensure the target folder exists
    os.makedirs('../reports', exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("Ensemble Model Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Runtime: {results['runtime']:.2f} seconds\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(results['confusion_matrix']) + "\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'] + "\n")
    
    print(f"Evaluation results saved to {filepath}")

# Example workflow
file_path = '../data/processed/spam_processed.csv'
df = load_data(file_path)
df = df.dropna()
X, y = extract_features_and_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Your train-test data
results = ensemble_model(X_train, y_train, X_test, y_test)

# Define the output file path for the evaluation results
output_filepath = '../reports/ensemble_model_evaluation.txt'
save_evaluation_results(results, output_filepath)