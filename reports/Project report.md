# **Spam Email Classification: Data Science/ML Report**

## **1. Project Overview**
This project focuses on building a spam email classification system using machine learning techniques. The goal was to process email data, train models to classify emails as spam or non-spam, and deploy an interactive app that allows users to classify their emails. Additionally, the app incorporates options for users to set custom preferences for what constitutes non-spam emails, adding a personalized touch to the classification process.

---

## **2. Objectives**
- Develop a robust machine learning pipeline to classify spam emails.
- Enable real-time predictions through a user-friendly interface.
- Allow users to upload datasets, input email text, or link their email accounts for classification.
- Provide options for users to set preferences for non-spam emails.
- Deploy the solution in a professional, scalable, and user-accessible manner.

---

## **3. Methodology**

### **3.1 Data Preprocessing**
1. **Dataset**:
   - The primary dataset was a CSV file containing email text and labels (`spam` or `ham`).
   - Additional input formats included:
     - Raw email text.
     - Email link for real-time extraction.

2. **Steps in Preprocessing**:
   - Converted text to lowercase.
   - Removed punctuation and special characters.
   - Tokenized text using NLTK’s `word_tokenize`.
   - Removed English stopwords.
   - Rejoined tokens into processed strings for further analysis.
   - Saved the cleaned dataset to the `../data/processed` folder.

### **3.2 Feature Extraction**
   - Extracted features using Term Frequency-Inverse Document Frequency (TF-IDF).
   - Split the dataset into training and testing sets using an 80-20 split.

### **3.3 Model Training**
   - Trained the following models using Scikit-learn:
     - Logistic Regression
     - Naive Bayes
     - Support Vector Machine (SVM)
   - Combined models using a VotingClassifier for ensemble learning.
   - Saved trained models to the `../models` folder for future use.

### **3.4 Evaluation**
   - Used the following metrics for evaluation:
     - Accuracy
     - Confusion Matrix
     - Classification Report (Precision, Recall, F1-score)
   - Visualized results using Seaborn and Matplotlib, saving all plots to the `../reports/images` folder.

### **3.5 Deployment**
   - Developed a Streamlit-based application with three versions:
     1. **Base App**: Demonstrates basic functionality using preprocessed data.
     2. **CSV App**: Accepts CSV datasets and classifies email data.
     3. **Email App**: Allows users to input email text or link an email account, providing personalization options.

---

## **4. Results**
- **Model Performance**:
  - Logistic Regression: 95% accuracy.
  - Naive Bayes: 93% accuracy.
  - SVM: 94% accuracy.
  - Ensemble VotingClassifier: 96% accuracy.

- **Visualizations**:
  - Model accuracy comparison bar chart.
  - Heatmaps of confusion matrices for individual models.

- **Deployment**:
  - Interactive app deployed via Streamlit.
  - User-friendly interface for personalized email classification.

---

## **5. Key Features of the Application**
1. Upload a dataset (CSV) and classify email data.
2. Input raw email text for classification.
3. Link email accounts for real-time classification.
4. Customize preferences for defining non-spam emails.
5. Access visualizations of model performance and classification results.

---

## **6. Skills Utilized**
- Python Programming
- Natural Language Processing (NLP)
- Machine Learning (Logistic Regression, Naive Bayes, SVM, Ensemble Models)
- Data Preprocessing (Pandas, NLTK, Regex)
- Feature Engineering (TF-IDF)
- Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
- Data Visualization (Matplotlib, Seaborn)
- Web App Development (Streamlit)
- Software Design and Pipeline Building

---

## **7. Challenges and Solutions**
1. **Handling Missing or Irregular Data**:
   - Used imputation and cleaning techniques to address missing values and irregularities.

2. **Balancing Model Performance**:
   - Improved performance using ensemble techniques.

3. **User Preferences in Classification**:
   - Integrated user-defined preferences into the email classification logic.

4. **Deploying a Scalable App**:
   - Designed the app to handle various input formats and user scenarios.

---

## **8. Conclusion**
This project demonstrates the power of combining machine learning and user-centric design to address a common problem: spam email detection. By creating a pipeline for data preprocessing, feature extraction, and model evaluation, we ensured robust and accurate predictions. The deployment of an interactive Streamlit app further highlights the practical application of this solution.

---

## **9. Future Work**
1. Incorporate additional features like email metadata (e.g., sender domain).
2. Expand user preferences to include categories of spam.
3. Add advanced NLP techniques, such as deep learning models (e.g., BERT).
4. Deploy the app as a web service for broader accessibility.

