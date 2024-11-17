# -*- coding: utf-8 -*-
"""MailGuard.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eDBWVy4l7B6fLsqymDKPjA1egGoR3PMf

# **MailGuard: Spam Detection System**


*MailGuard is a spam email detection system built using machine learning models to classify emails as either spam or ham (legitimate). It leverages natural language processing (NLP) techniques, such as TF-IDF vectorization, and a variety of machine learning models, including Logistic Regression, Random Forest, and XGBoost. The project aims to provide an efficient and scalable solution for automatically classifying email messages.*

# **Spam Mail Detector**

**Programming Language**
* Python

**Libraries and Frameworks**
* Pandas: For data manipulation and analysis
* NumPy: For numerical computations
* Scikit-learn:
  * train_test_split: For splitting data into training and testing sets.
  * TfidfVectorizer: For feature extraction from text data.
  * LogisticRegression, RandomForestClassifier: For machine learning models.
  * GridSearchCV: For hyperparameter tuning.
  * accuracy_score, classification_report, confusion_matrix: For model evaluation.
* XGBoost: A high-performance machine learning model used for classification tasks.
* re (Regular Expressions): For preprocessing and cleaning email data.

**Machine Learning Algorithms**
* Logistic Regression: A simple linear model for binary classification (ham or spam).
* Random Forest Classifier: An ensemble learning method that can handle complex relationships in data.
* XGBoost Classifier: A powerful boosting algorithm for high performance, often used in Kaggle competitions for classification tasks.

**Data Processing and Feature Engineering**
* TF-IDF Vectorization: Converts raw email text into numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF), with the option for n-grams (unigrams and bigrams) to capture word pairs.

**Hyperparameter Tuning**
* GridSearchCV: For tuning the hyperparameters of models like XGBoost to get the best results.

**Data Storage**
* CSV Files: Data is stored in CSV files (e.g., mail_data.csv), which is loaded and processed using Pandas.

**Evaluation Metrics**
* Accuracy: Measures the percentage of correct predictions (spam vs. ham).
* Classification Report: Provides precision, recall, and F1-score for each class (spam/ham).
* Confusion Matrix: Helps visualize the performance of classification models.

**Environment**
* Jupyter Notebook or IDE (like VS code or PyCharm):  For developing and testing your models interactively.
* Google Colab: If you're working on a cloud-based environment for easy access to resources.

**Importing the Libraries**
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

"""**Data Loading and Preprocessing**"""

def load_and_preprocess_data(file_path):
    raw_mail_data = pd.read_csv(file_path)
    mail_data = raw_mail_data.fillna('')
    mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})
    mail_data['Message'] = mail_data['Message'].apply(clean_text)
    return mail_data

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

file_path = '/content/mail_data.csv'
mail_data = load_and_preprocess_data(file_path)

X = mail_data['Message']
Y = mail_data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

"""**Future Engineering**"""

def extract_features(X_train, X_test):
    feature_extraction = TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)
    return X_train_features, X_test_features, feature_extraction

X_train_features, X_test_features, feature_extraction = extract_features(X_train, X_test)

"""**Model Defination**"""

def define_models():
    lr_model = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.5, random_state=42)
    rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)

    return {
        "Logistic Regression": lr_model,
        "XGBoost": xgb_model,
        "Random Forest": rf_model
    }

models = define_models()

"""**Hyperparameter Tuning**"""

def tune_xgboost(X_train_features, Y_train):
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [100, 200]
    }
    grid_search = GridSearchCV(estimator=models["XGBoost"], param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    grid_search.fit(X_train_features, Y_train)
    return grid_search.best_estimator_

xgb_best_model = tune_xgboost(X_train_features, Y_train)
models["XGBoost (Tuned)"] = xgb_best_model

"""**Train and Evaluate Models**"""

def train_and_evaluate_models(models, X_train_features, Y_train, X_test_features, Y_test):
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train_features, Y_train)

        train_predictions = model.predict(X_train_features)
        test_predictions = model.predict(X_test_features)

        train_accuracy = accuracy_score(Y_train, train_predictions)
        test_accuracy = accuracy_score(Y_test, test_predictions)

        print(f"{model_name} - Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"{model_name} - Testing Accuracy: {test_accuracy * 100:.2f}%")
        print(f"{model_name} - Classification Report:\n", classification_report(Y_test, test_predictions, target_names=['Spam', 'Ham']))
        print(f"{model_name} - Confusion Matrix:\n", confusion_matrix(Y_test, test_predictions))

train_and_evaluate_models(models, X_train_features, Y_train, X_test_features, Y_test)

"""**Custom Input Prediction**"""

def predict_custom_input(models, input_mail, feature_extraction):
    input_mail_cleaned = [clean_text(mail) for mail in input_mail]
    input_mail_features = feature_extraction.transform(input_mail_cleaned)

    print("\nCustom Input Prediction:")
    for model_name, model in models.items():
        prediction = model.predict(input_mail_features)
        result = "Ham mail" if prediction[0] == 1 else "Spam mail"
        print(f"{model_name}: {result}")

input_mail = ["Congratulations! You've won a $1000 gift card. Click here to claim now!"]
predict_custom_input(models, input_mail, feature_extraction)