# Yelp rating predictor

### Data
https://www.kaggle.com/c/yelpratingprediction/data

### Reqs
+ python3, requests, numpy, pandas, scikit-learn
+ csv data: business.csv, users.csv, train_review.csv, validate_queries.csv, test_queries.csv

### Goal
Based on business.csv, users.csv, train_review.csv, predict users rating in test_queries.csv 
while minimizing root mean square error

### Best result
1.04389 RMSE with random forest

### Outline
+ Clean business.csv, users.csv (impute missing data while remove insignificant, unusable, 
highly missing attributes, quanitify non numerical attributes)
+ Merge the 2 cleaned csv with each of train_review.csv and test_queries.csv for training 
data and test data respectively
+ Use various models (linear regression, random forest, nn) to predict the "stars" column
