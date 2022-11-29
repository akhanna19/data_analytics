# Project - Bank Direct Marketing: A Data Science Project


## Overview of the Project

This is a marketing data science project. The objective of this project is to build a predictive model that will help predict whether an existing customer, given his or her relevant data, will subscribe to a term deposit scheme offered by the bank. To achieve this objective, I will first model five classifiers and will then select and deploy the best performing model.

The five classifiers that I will train are - 

  1. Naive Bayes
  2. Logistic Regression
  3. K-Nearest Neighbors
  4. Random Forest
  5. XGBoost

Each model will attempt to predict attempt to predict whether a client will subscribe to a term deposit scheme (labeled 1) or not (labeled 0).


## Overview of the Dataset

This direct marketing dataset is of a Portuguese banking institution. I got this dataset from UC Irvine's Machine Learning Repository [URL = https://archive.ics.uci.edu/ml/datasets/bank+marketing].

The dataset contains 45211 records and 17 variables. The predictors can be grouped into three broad categories - 

  1. Client's personal information
  2. Client's financial information
  3. Previous interactions between the bank and that client


## Research Questions

I wanted to look at three research questions - 

  1. What are the key features that help predict whether a client will subscribe to a term deposit?

  2. The marketing campaign was conducted using phone calls. Often the bank employees contact a customer several times to get them to subscribe to the term deposit. I want to find out the proportion of successful calls?

  3. Does a client’s educational status and job type have an impact on his or her decision-making?


## Model Training Results

* Model 1 = Naive Bayes:
  * Precision for Class 1 (subscribed to the scheme) = 0.72
  * Recall for Class 1 (subscribed to the scheme) = 0.46
  * AUC Score = 0.638
  
* Model 2 = Logistic Regression:
  * Precision for Class 1 (subscribed to the scheme) = 0.69
  * Recall for Class 1 (subscribed to the scheme) = 0.60
  * AUC Score = 0.662

* Model 3 = K-Nearest Neighbors:
  * Precision for Class 1 (subscribed to the scheme) = 0.81
  * Recall for Class 1 (subscribed to the scheme) = 0.94
  * AUC Score = 0.861

* Model 4 = Random Forest:
  * Precision for Class 1 (subscribed to the scheme) = 0.93
  * Recall for Class 1 (subscribed to the scheme) = 0.99
  * AUC Score = 0.957 

* Model 5 = XGBoost:
  * Precision for Class 1 (subscribed to the scheme) = 0.77
  * Recall for Class 1 (subscribed to the scheme) = 0.66
  * AUC Score = 0.731


## Final Model

The best performing model, in terms of all three performance metrics - Precision, Recall, & AUC Score - is the Random Forest. The random forest model has a phenomenal recall score of 99% (i.e., out of all original positives, the model was able to recall 99% of them accurately].

However, in spite of such great performance, I believe the model has overfitted on the training dataset.

As a result, my final choice of model is the K-Nearest Neighbor model (number of neighbors = 4). Although one of the biggest drawbacks of this model is it's high training time, I will still go with this one as it gives a great recall value (94%) and a decent AUC score of 86%.