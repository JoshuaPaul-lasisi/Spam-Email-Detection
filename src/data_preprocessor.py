import os
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def load_data(filepath):
    """Load dataset from the provided file path."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the data by converting to lowercase, removing punctuation, tokenizing, and removing stopwords."""
    df['Spam'] = df['Category'].map({'spam': 1, 'ham': 0})
    df['Message'] = df['Message'].str.lower().str.replace(f"[{string.punctuation}]", " ", regex=True)
    df['Message'] = df['Message'].apply(word_tokenize)
    
    stop_words = set(stopwords.words('english'))
    df['Message'] = df['Message'].apply(lambda x: [word for word in x if word not in stop_words])
    df['Message'] = df['Message'].apply(lambda x: ' '.join(x))
    df = df.dropna()
    
    return df

def save_processed_data(df, output_filepath):
    """Save the preprocessed data to the specified output filepath."""
    # Ensure the target folder exists
    processed_folder = '../data/processed'
    os.makedirs(processed_folder, exist_ok=True)
    
    # Save the processed data to a CSV file
    df.to_csv(output_filepath, index=False)
    print(f"Processed data saved to {output_filepath}")

# Example workflow
filepath = '../data/raw/spam.csv'
spam_df = load_data(filepath)
spam_df = preprocess_data(spam_df)

# Define the output file path for the processed data
output_filepath = '../data/processed/spam_processed.csv'
save_processed_data(spam_df, output_filepath)