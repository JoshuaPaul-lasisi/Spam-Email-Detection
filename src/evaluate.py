import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_models(data_path, models_dir, reports_dir):
    # Load processed data
    df = pd.read_csv(data_path)
    
    # Load vectorizer
    vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
    
    X = vectorizer.transform(df['Message'])
    y = df['Spam']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load models and evaluate
    models = ['logistic_regression', 'naive_bayes', 'svm', 'ensemble_model']
    
    for model_name in models:
        model = joblib.load(os.path.join(models_dir, f'{model_name}.pkl'))
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy}")
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d')
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(os.path.join(reports_dir, f'{model_name}_conf_matrix.png'))
        plt.clf()
        
        # Classification Report
        report = classification_report(y_test, y_pred)
        with open(os.path.join(reports_dir, f'{model_name}_classification_report.txt'), 'w') as f:
            f.write(report)

if __name__ == "__main__":
    evaluate_models('../data/processed/spam_data_processed.csv', '../models/', '../reports/images/')