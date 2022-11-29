# Credit Risk Classification Model

## Overview of the Analysis

The purpose of this machine learning exercise is to create a model that can classify between healthy loan and high-risk loan as best as possible. To perform this classification task aimed at identifying the creditworthiness of borrowers, I use the Logistic Regression model. The dataset that I use to build the model is of historical lending activity from a peer-to-peer lending services. 

The label that the model will predict is "loan_status". This is a binary nominal variable that has only 2 classes, 0 (denoting healthy loan) and 1 (denoting high-risk loan). From my analysis and by virtue of the nature of the problem at hand, the target variable is heavily imbalanced. Using Pandas value_counts() function, I found that the dataset contains 75036 records for class 0 (healthy loan), which constitutes almost 97% of the entire dataset, and only 2500 records for class 1 (high-risk loan), which constitutes merely 3% of the dataset.

Becasue of the presence of imbalanced dataset, my analysis will be divided into two parts - 

* In the first part, I use the imbalanced dataset as is to create the Logistic Regression model.
* In the second part, I use the RandomOverSampler module from the imbalanced-learn package to randomly oversample the minority class (class 1, high-risk loan) by picking samples at random with replacement. I then use this oversampled or resampled data to create the Logistic Regression model. 

For each part of the analysis, I fit the Logistic Regression model to each dataset, respectively. I then then evaluate the performance of each model using the Accuracy Score, the Confusion Matrix, and finally the Classification Report. 


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Accuracy Score = 0.95
  * Precision for Class 0 (healthy loan) = 1.00
  * Precision for Class 1 (high-risk loan) = 0.85
  * Recall for Class 0 (healthy loan) = 0.99
  * Recall for Class 1 (high-risk loan) = 0.91

* Machine Learning Model 2:
  * Accuracy Score = 0.99
  * Precision for Class 0 (healthy loan) = 1.00
  * Precision for Class 1 (high-risk loan) = 0.84
  * Recall for Class 0 (healthy loan) = 0.99
  * Recall for Class 1 (high-risk loan) = 0.99

## Summary

### Model 1 - 

Looking at the Confusion Matrix, we see that the Logistic Regression Model predicted the correct labels 19226 records (18663 for True Positive + 563 for True Negative). 

Looking at the Classification Report, we see that the precision and the recall for the class 0 (healthy loan) and class 1 (high-risk loan) is 1.00 and 0.99, respectively. Both precision and recall are much better for class 0 than for class 1.

The precision for class 0 is 1.00, which means that out of all the times the model predicted a 0, 100% of those predictions were correct. By contrast, out of all the times that the model predicted a value of 1, only 85% of those predictions were correct. A possible reason for 100% precision for class 0 could be because our dataset is heavily imbalanced toward class 0. That is, almost 97% of our records belong to class 0 and merely 3% belong to class 1. As a result, it is easier for the model to get predictions for class 0 correct 100% of the time.

On the other hand, the recall for class 0 is 0.99, which means that out of all the original values of class 0, the model was able to correctly recall class = 0 99% of the time. This fares better than recall value of 0.91 for class 1, which means that out of all original values of class 1, the model was able to correctly recall class = 1 91% of the time.

Additionally, the accuracy of the model is 95%. It is again important to note that our dataset is heavily imbalanced. As a result, the accuracy score doesn't give an accurate picture of the model's performance.


### Model 2 -

Looking at the Confusion Matrix, we see that the Logistic Regression Model predicted the correct labels 19264 records (18649 for True Positive + 615 for True Negative).

Looking at the Classification Report, we see that the precision and the recall for the class 0 (healthy loan) and class 1 (high-risk loan) is 1.00 and 0.99, respectively. Precision is much better for class 0 than for class 1, whereas Recall is the same for both.

The precision for class 0 is 1.00, which means that out of all the times the model predicted a 0, 100% of those predictions were correct. By contrast, out of all the times that the model predicted a value of 1, only 84% of those predictions were correct. Because the oversampled data is not imbalanced, precision score of 100% and 84% for class 0 and class 1, respectively, are quite good.

On the other hand, the recall for both class 0 is 0.99, which means that out of all the original values of class 0, the model was able to correctly recall class = 0 99% of the time. The recall value for class 1 is also the same; out of all original values of class 1, the model was able to correctly recall class = 1 99% of the time.

Additionally, the accuracy of the model with equally balanced data is almost 99%, which is again a solid score.

Looking at these numbers, we can conclude that the Logistic Regression model fits well with the oversampled data.


### Final choice - 

For the given problem, we want a model that can accurately predict the high-risk loan (class 1). On the face of it, the raw metrics for model 2 fare slightly better than the raw metrics for model 1. However, on closer look we see that the precision score for class 1 given by model 2 (0.84) is slightly less than that of model 1. Precision score basically indicates that out of all predicted class 1 labels, how many were accurately captured by the model. A score of 0.84 (given by model 2) indicates that of all class 1 predictions, only 84% of them were correct (i.e., only 84% of them truly belonged to class 1). Overall, this is quite less. Similarly, a score of 0.85 (given by model 1) is no better.

Because of this reason, I do not recommend any of the models.