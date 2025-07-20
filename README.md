
# Sentiment Analysis with Python
This repository contains a Jupyter Notebook (centimentAnalysis.ipynb) that demonstrates a complete workflow for performing sentiment analysis on text data. The project classifies text reviews into two categories: positive (label 1) and negative (label 0).

**Project Overview**:
Sentiment analysis is a Natural Language Processing (NLP) task that involves determining the emotional tone behind a piece of text. This project builds a machine learning model to automate this process, which can be highly valuable for tasks like customer feedback analysis, social media monitoring, and review summarization.

The notebook covers the following key stages:

**Data Loading & Initial Exploration**: Loading the dataset and understanding its basic structure.

**Sentiment Distribution Analysis**: Visualizing the balance of positive and negative sentiments in the dataset.

*Text Preprocessing**: Cleaning and normalizing raw text data to prepare it for machine learning.

**Feature Engineering**: Converting textual data into numerical features that a machine learning model can process.

**Model Training**: Training a Logistic Regression classifier (with cross-validation) on the prepared data.

**Model Evaluation**: Assessing the performance of the trained model using standard metrics.

Key Features & Functionality
Data Loading: Reads text data and labels from a CSV file.

Sentiment Visualization: Generates a pie chart to show the proportion of positive vs. negative reviews.

Text Cleaning Pipeline: Includes steps to remove HTML tags, handle emojis, convert text to lowercase, and remove punctuation.

Tokenization & Stemming: Breaks text into words and reduces them to their root forms for consistency.

TF-IDF Feature Extraction: Transforms text data into numerical TF-IDF vectors, capturing word importance.

Train-Test Split: Divides the dataset into training and testing sets for robust model evaluation.

Logistic Regression Model: Implements a Logistic Regression classifier for sentiment prediction, utilizing built-in cross-validation for better generalization.

Accuracy Measurement: Calculates and displays the overall accuracy of the trained model on unseen data.

Word Cloud Generation: Visualizes the most frequent words in both positive and negative reviews after preprocessing.

Technologies Used
Python: The core programming language for the project.

Pandas: For data manipulation and analysis (DataFrame operations, CSV reading).

NumPy: Underpins numerical operations in data processing and machine learning.

Matplotlib: For creating static, interactive, and animated visualizations (pie charts, word clouds).

Scikit-learn (sklearn): A comprehensive machine learning library.

CountVectorizer (imported, though TfidfVectorizer is used for feature extraction)

TfidfVectorizer (for TF-IDF feature extraction)

train_test_split (for data splitting)

LogisticRegressionCV (for model training with cross-validation)

metrics (for model evaluation, specifically accuracy_score)

NLTK (Natural Language Toolkit): For advanced text processing.

PorterStemmer (for stemming words)

stopwords (for removing common words)

re (Regular Expressions): Python's built-in module for pattern matching in text.

WordCloud: For generating visual word cloud representations.

Important Concepts & Techniques Applied
Natural Language Processing (NLP): The field of AI that deals with the interaction between computers and human language.

Sentiment Analysis: A subfield of NLP focused on identifying and extracting subjective information from text.

Text Preprocessing: Essential steps to clean and standardize text data (e.g., lowercasing, HTML tag removal, punctuation removal, emoji handling).

Tokenization: Breaking text into individual units (words or subwords).

Stemming: Reducing words to their root form (e.g., "running" -> "run").

Stop Word Removal: Eliminating common, less informative words.

Feature Engineering for Text: Transforming raw text into numerical features.

TF-IDF (Term Frequency-Inverse Document Frequency): A statistical measure reflecting the importance of a word in a document relative to a collection of documents.

Supervised Machine Learning: Training a model on labeled data (text -> label).

Classification: Predicting a categorical output (positive or negative sentiment).

Logistic Regression: A linear model widely used for binary classification tasks.

Cross-Validation: A robust technique for evaluating model performance and selecting optimal hyperparameters by training and testing on different subsets of the data.

Model Evaluation: Using metrics like Accuracy to quantify how well the model performs.

Data Visualization: Using pie charts and word clouds to gain insights from the data.

How to Run the Project
Clone the repository:

git clone <repository_url>
cd sentiment-analysis-project

Ensure you have Python installed (preferably Python 3.7+).

Install the required libraries:

pip install pandas numpy matplotlib scikit-learn nltk wordcloud

Download NLTK data: Open a Python interpreter or run the following in a script:

import nltk
nltk.download('stopwords')
nltk.download('wordnet') # If you decide to use lemmatization
nltk.download('omw-1.4') # If you decide to use lemmatization

Place your Train.csv dataset in the root directory of the project.

Open the Jupyter Notebook:

jupyter notebook centimentAnalysis.ipynb

Run the cells sequentially within the Jupyter environment.

Achieved Accuracy
The current Logistic Regression model achieves an accuracy of approximately 89.06% on the test set.

Possible Improvements
Advanced Text Preprocessing:

Implement Lemmatization instead of stemming for potentially better root word representation.

Experiment with negation handling (e.g., "not good" as a single token).

Explore custom stop words tailored to the dataset's domain.

N-grams: Include bigrams and trigrams in the TF-IDF feature extraction to capture contextual meaning (e.g., ngram_range=(1, 2) or (1, 3) in TfidfVectorizer).

Hyperparameter Tuning: Perform a more exhaustive Grid Search or Randomized Search for LogisticRegressionCV or LogisticRegression to find optimal C, penalty, and solver parameters.

Class Imbalance Handling: If the dataset is imbalanced, use class_weight='balanced' in LogisticRegressionCV or explore oversampling (e.g., SMOTE) or undersampling techniques.

Alternative Models:

Support Vector Classifier (SVC): Often performs very well on text data.

Naive Bayes (MultinomialNB): A strong baseline for text classification.

Ensemble Methods: Such as RandomForestClassifier or GradientBoostingClassifier (e.g., LightGBM, XGBoost).

Deep Learning: For very large datasets, consider deep learning models like Recurrent Neural Networks (RNNs) or Transformer-based models (e.g., BERT, RoBERTa) for potentially higher accuracy, though this requires more computational resources and a different skill set.

More Evaluation Metrics: Beyond accuracy, consider precision, recall, F1-score, and ROC-AUC for a more comprehensive understanding of model performance, especially with imbalanced datasets.

