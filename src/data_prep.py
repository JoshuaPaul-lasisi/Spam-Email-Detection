import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def load_and_preprocess_data(input_path, output_path):
    # Load the data
    df = pd.read_csv(input_path)

    # Preprocess data
    df['Spam'] = df['Category'].map({'spam': 1, 'ham': 0})
    df['Message'] = df['Message'].str.lower().str.replace(f"[{string.punctuation}]", " ", regex=True)
    df['Message'] = df['Message'].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    df['Message'] = df['Message'].apply(lambda x: [word for word in x if word not in stop_words])
    df['Message'] = df['Message'].apply(lambda x: ' '.join(x))

    # Save the preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    load_and_preprocess_data('data/raw/spam_data.csv', 'data/processed/spam_data_processed.csv')