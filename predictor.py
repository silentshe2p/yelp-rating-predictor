#!/usr/bin/python3

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from dataCleaning import cleanBusinessData, cleanUsersData
from dataCombining import combine_data

business_csv = "business.csv"
cleaned_business_csv = "business_cl.csv"
users_csv = "users.csv"
cleaned_users_csv = "users_cl.csv"
train_reviews_csv = "train_reviews.csv"
train_reviews_combined_csv = "train_reviews_combined.csv"
validate_queries_csv = "validate_queries.csv"
validate_queries_combined_csv = "validate_queries_combined.csv"
test_queries_csv = "test_queries.csv"
test_queries_combined_csv = "test_queries_combined.csv"
submission_csv = "submission.csv"
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

def get_df(file, y_name):
    df = pd.read_csv(file, index_col=False)
    y = df[y_name]
    X = df.drop(y_name, axis=1)
    return X, y

def compute_rmse(predicted_y, y):
    return np.sqrt(np.sum((predicted_y - y)**2)/y.shape[0])

def compute_accuracy(predicted_y, y):
    return np.sum(predicted_y == y)/y.shape[0]

class StarPredictor(object):
    def __init__(self, reviews_csv, train_test_reviews_csv, test_reviews_csv):
        self.train_X, self.train_y = get_df(reviews_csv, "review_stars")
        self.train_X = self.train_X.drop(["Unnamed: 0"], axis=1, errors="ignore")
        
        self.train_test_X, self.train_test_y = get_df(train_test_reviews_csv, 
                                                      "review_stars")
        self.train_test_X = self.train_test_X.drop(
                                ["Unnamed: 0", "Unnamed: 0_x", "Unnamed: 0_y"], 
                                axis=1, errors="ignore")
        
        self.test_X = pd.read_csv(test_reviews_csv)
        self.test_X = self.test_X.drop(
                                ["Unnamed: 0", "Unnamed: 0_x", "Unnamed: 0_y"], 
                                axis=1, errors="ignore")

    def linear_regression(self):
        lr = LinearRegression()
        lr.fit(self.train_X, self.train_y)
        print("# Training RMSE:", compute_rmse(lr.predict(self.train_X), self.train_y))

        # Predict train test
        train_test_pred = lr.predict(self.train_test_X)
        tt_rmse = compute_rmse(train_test_pred, self.train_test_y)
        print("# Training test RMSE:", tt_rmse)
        print("# Training test accuracy:", compute_accuracy(np.rint(train_test_pred), 
                                                              self.train_test_y))
        
        # Predict test
        test_pred = lr.predict(self.test_X)
        return tt_rmse, test_pred
    
    def random_forest(self, param_grid, verbose=0, save_model=True):
        rf = RandomForestRegressor()

        # Grid search with 3-fold cross validation, all processors
        gs = GridSearchCV(estimator = rf, param_grid = param_grid, n_jobs=-1,
                          cv = 3, scoring = mse_scorer, verbose = 1)
        gs = gs.fit(self.train_X, self.train_y)

        # Train with the best estimator
        print("# Best rf hyperparams:", gs.best_params_)
        rf = gs.best_estimator_
        rf.fit(self.train_X, self.train_y)
        
        # Predict train test
        train_test_pred = rf.predict(self.train_test_X)
        tt_rmse = compute_rmse(train_test_pred, self.train_test_y)
        print("# Training test RMSE:", tt_rmse)
        print("# Training test accuracy:", compute_accuracy(np.rint(train_test_pred), 
                                                              self.train_test_y))
              
        # Feature importances
        if verbose == 1:  
            importances = list(rf.feature_importances_)
            # Feature-importance tuple
            feature_importances = [(feature, importance) 
                for feature, importance in zip(self.train_test_X, importances)]
            # Sort by importance
            feature_importances = sorted(feature_importances, 
                                         key = lambda x: x[1], reverse = True)
            for pair in feature_importances:
                print('# Feature: {:25} Importance: {}'.format(*pair))
                  
        # Predict test
        test_pred = rf.predict(self.test_X)
        
        # Save model to pickle
        if save_model:
            model_pickle = ("rf" + "_" + str(gs.best_params_["n_estimators"]) 
                                 + "_" + str(gs.best_params_["max_depth"])
                                 + "_" + str(gs.best_params_["min_samples_split"])
                                 + "_" + str(gs.best_params_["min_samples_leaf"])
                                 + ".pickle")
            with open(model_pickle, 'wb') as handle:
                pickle.dump(rf, handle)
                
        return tt_rmse, test_pred
    
    def nn(self, param_grid):      
        mlpr = MLPRegressor()
        # Grid search with 3-fold cross validation, all processors
        gs = GridSearchCV(estimator = mlpr, param_grid = param_grid, n_jobs=3,
                          cv = 3, scoring = mse_scorer, verbose = 2)
        gs = gs.fit(self.train_X, self.train_y)
        
        # Train with the best estimator
        print("# Best nn hyperparams:", gs.best_params_)
        mlpr = gs.best_estimator_
        mlpr.fit(self.train_X, self.train_y)

        # Predict train test
        train_test_pred = mlpr.predict(self.train_test_X)
        tt_rmse = compute_rmse(train_test_pred, self.train_test_y)
        print("# Training test RMSE:", tt_rmse)
        print("# Training test accuracy:", compute_accuracy(np.rint(train_test_pred), 
                                                              self.train_test_y))
        # Predict test
        test_pred = mlpr.predict(self.test_X)
        return tt_rmse, test_pred
    
if __name__ == '__main__':
    print("### Clean business data")
    cleanUsersData(users_csv, cleaned_users_csv)
    print("### Clean user data")
    cleanBusinessData(business_csv, cleaned_business_csv)
    print()
    print("### Combine data")
    combine_data(cleaned_users_csv, cleaned_business_csv, train_reviews_csv, 
                                        validate_queries_csv, test_queries_csv)
    print()
    print("### Predict")
    predictor = StarPredictor(train_reviews_combined_csv,
                              validate_queries_combined_csv,
                              test_queries_combined_csv)
    results = []
    train_test_rmse = []
    models = ["Linear regression", "Random forest", "NN"]

    print("## Linear regression model")
    lr_rmse, lr_result = predictor.linear_regression()
    results.append(lr_result)
    train_test_rmse.append(lr_rmse)
    print("###################################################################")
    
    print("## Random forest model")
    # Hyperparam tuning
    rf_grid = {'n_estimators': [46],
                       'max_features': ['auto'],
                       'max_depth': [9],
                       'min_samples_split': [8],
                       'min_samples_leaf': [6]}
    
    # Uncomment this part for full parameters grid search
#    # RF number of trees
#    n_estimators = [i for i in range(20, 81)]
#    # Number of features to consider at every split
#    max_features = ['auto', 'sqrt']
#    max_depth = [i for i in range(1, 21)]
#    max_depth.append(None)
#    min_samples_split = [i for i in range(2, 11)]
#    min_samples_leaf = [i for i in range(1, 11)]
#
#    rf_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf}

    rf_rmse, rf_result = predictor.random_forest(rf_grid)
    results.append(rf_result)
    train_test_rmse.append(rf_rmse)
    print("###################################################################")
    
    print("## NN model")
    # Hyperparam tuning
    nn_grid = {'solver': ["adam"],
               'alpha': [1e-5],
               'learning_rate': ["adaptive"],
               'activation': ["tanh"],
               'hidden_layer_sizes': [(3,1,1)]}
    
    # Uncomment this part for full parameters grid search
#    first_lay = [i for i in range(3, 101, 1)]
#    second_lay = [1, 2, 3, 4, 5]
#    no_second_lay = [(f,) for f in first_lay]
#    with_second_lay = [(f, s) for f in first_lay for s in second_lay]
#    hidden_lay = no_second_lay + with_second_lay
#    nn_grid = {'solver': ["adam"],
#               'alpha': [1e-5],
#               'learning_rate': ["constant", "invscaling", "adaptive"],
#               'activation': ["relu", "tanh"],
#               'hidden_layer_sizes': hidden_lay}
    
    nn_rmse, nn_result = predictor.nn(nn_grid)
    results.append(nn_result)
    train_test_rmse.append(nn_rmse)
    print("###################################################################")
          
    best_rmse = np.argmin(np.array(train_test_rmse))
    print(f"## {models[best_rmse]} model result was chosen")
    stars_df = pd.DataFrame(data=results[best_rmse], columns=["stars"])
    stars_df.index.name = "index"
    stars_df.to_csv(submission_csv, index="index")    
    print("## Saved prediction to", submission_csv)
    print("### All done")
    