# False News Detection System

## Overview
This project implements a model for detecting unreliable or false news articles. Two primary machine learning algorithms are utilized for classification: **Logistic Regression** and **Support Vector Machines (SVM)**. An experimental analysis compares the performance of these models in predicting whether news articles are true or false. The data preprocessing, model training, testing, and evaluation are conducted using Python libraries such as Scikit-Learn and Flask.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Classification Models](#classification-models)
  - [Logistic Regression](#logistic-regression)
  - [Support Vector Machines](#support-vector-machines)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

## Data
The dataset for this project is sourced from [Kaggle](https://www.kaggle.com), consisting of articles labeled as either **true** or **false**. Each article includes:
- **Author**: The writer of the article.
- **Title**: The title of the news article.
- **Content**: The full text of the article.
- **Label**: A binary label, where `1` represents true news and `0` represents false news.

## Preprocessing
Before applying the machine learning algorithms, the data is preprocessed using Python libraries:
- The **author** and **title** columns are merged to form a new text feature.
- The **label** column is separated as the target variable.
- Text cleaning operations include:
  - Removal of punctuation marks.
  - Conversion of uppercase letters to lowercase.
  - Elimination of extra spaces.

This preprocessing ensures that the text data is in a format suitable for the classification models.

## Classification Models

### Logistic Regression
Logistic regression is well-suited for this binary classification problem, producing output values between `0` and `1`. Articles predicted with values close to `0` are classified as **false**, while those close to `1` are classified as **true**.

### Support Vector Machines (SVM)
The SVM algorithm operates similarly, but with output values close to `-1` for **false** and `1` for **true**. Due to the relatively small size of the dataset, SVM scales well for this use case.

## Methodology
The following steps were followed to build and evaluate the models:
1. **Train-Test Split**: The dataset was randomly divided into training and testing sets using the `train_test_split()` function from the Scikit-Learn library.
2. **Text Vectorization**: Text data was transformed into vector form using the `TfidfVectorizer` model from Scikit-Learn.
3. **Model Training and Testing**: Both Logistic Regression and SVM models were implemented. Tests were conducted on both training and test sets to detect overfitting issues.
4. **Time Evaluation**: The `time` library was used to measure the duration of model training and the time required to make predictions on new text.
5. **Web Interface**: A simple web application was created using the Flask library, allowing users to input news text via a webpage. The text is preprocessed and passed through the trained model to assess its authenticity.

## Results
The results of the experimental analysis, including the accuracy of both Logistic Regression and SVM on the test dataset, will be discussed here.

## How to Run
To run this project locally, follow these steps:

1
