from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features_and_target(df):
    """Extract features and target variable for model training."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Message'])
    y = df['Spam']
    return X, y