# **MailGuard: Spam Detection System**

MailGuard is a spam email detection system built using machine learning models to classify emails as either spam or ham (legitimate). It leverages natural language processing (NLP) techniques, such as TF-IDF vectorization, and a variety of machine learning models, including Logistic Regression, Random Forest, and XGBoost. The project aims to provide an efficient and scalable solution for automatically classifying email messages.

## **Project Overview**
MailGuard is a machine learning-based spam email detection system designed to automatically classify emails as spam or legitimate (ham). It utilizes Natural Language Processing (NLP) techniques for text processing and several machine learning models such as Logistic Regression, Random Forest, and XGBoost to classify email messages accurately. This project can be used to filter out unwanted emails, helping to improve productivity and email security.

### **Programming Language**
* Python

### **Libraries and Frameworks**
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

### **Machine Learning Algorithms**
* Logistic Regression: A simple linear model for binary classification (ham or spam).
* Random Forest Classifier: An ensemble learning method that can handle complex relationships in data.
* XGBoost Classifier: A powerful boosting algorithm for high performance, often used in Kaggle competitions for classification tasks.

### **Data Processing and Feature Engineering**
* TF-IDF Vectorization: Converts raw email text into numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF), with the option for n-grams (unigrams and bigrams) to capture word pairs.

### **Hyperparameter Tuning**
* GridSearchCV: For tuning the hyperparameters of models like XGBoost to get the best results.

### **Data Storage**
* CSV Files: Data is stored in CSV files (e.g., mail_data.csv), which is loaded and processed using Pandas.

### **Evaluation Metrics**
The performance of the models is evaluated using:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A matrix showing true positives, false positives, true negatives, and false negatives.

### **Environment**
* Jupyter Notebook or IDE (like VS code or PyCharm):  For developing and testing your models interactively.
* Google Colab: If you're working on a cloud-based environment for easy access to resources.

## **Model Accuracy**
The project evaluates multiple machine learning models to classify emails as spam or ham. After training and testing each model, the following accuracies were obtained on the test set (unseen data):

* Logistic Regression:
  * Training Accuracy: 97.30%
  * Testing Accuracy: 96.15%
    
* Random Forest Classifier:
  * Training Accuracy: 99.90%
  * Testing Accuracy: 97.50%
    
* XGBoost Classifier (Tuned):
  * Training Accuracy: 99.95%
  * Testing Accuracy: 98.30%

## **Installation Instructions**

1. Clone the repository:
   ```bash
   git clone https://github.com/Ayushpremrocks/MailGuard-Spam-Detection-System.git

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the script to train and test the model:
   ```bash
   python main.py

NOTE: If you're using Google Colab, simply just upload the CSV file and the run the cells.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
