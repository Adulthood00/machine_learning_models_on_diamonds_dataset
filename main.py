from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
import pandas as pd
from helper import *
from explore import *
import sys


diamonds=pd.read_csv("diamonds.csv")


explore_data(diamonds)
diamonds = diamonds.drop("Unnamed: 0", axis=1)

#More exploration with plots
barplot_against_price("color",diamonds)
barplot_against_price("cut",diamonds)
barplot_against_price("clarity", diamonds)

histogram(diamonds["carat"], "Carat Weight", "Frequency", "Distribution of Diamond Carat Weight")
histogram(diamonds["price"], "Price in USD", "Frequency", "Distribution of Diamond Price")

#check structure and correlation between data with plots

# Encode string data

data = diamonds.copy()

data["cut"] = encoder(data["cut"])
data["clarity"] = encoder(data["cut"])
data["color"] = encoder(data["cut"])

data.head()

scatter_matrix(data, figsize=(12, 8))
correlation(data)
sns.pairplot(data, x_vars=['carat',"x","y","z"], y_vars="price", size=7, aspect=0.7,kind="reg")

def main():
    """
    Trains different models and use them to predict test data and print the results
    :return:
    """
    #Using only the data with the highest correlation with the target
    data_final=pd.DataFrame([data["carat"],data["x"],data["y"],data["z"],data["price"]]).T
    print(data_final.head())

    #Create the predictors and the target

    predictors = data_final.drop("price", axis=1)
    target = data_final["price"].copy()

    predictors = standard_scaler(predictors)

    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20)

    #Create the models and measure their rmse

    create_model_and_measure_rmse(LinearRegression(), X_train, y_train, X_test, y_test)

    create_model_and_measure_rmse(SGDRegressor(max_iter=50,tol=None,penalty=None,eta0=0.1), X_train, y_train, X_test, y_test)

    create_model_and_measure_rmse(LinearSVR(epsilon=1.5), X_train, y_train, X_test, y_test)

    create_model_and_measure_rmse(BaggingRegressor(
    DecisionTreeRegressor(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
    ), X_train, y_train, X_test, y_test)

    create_model_and_measure_rmse(RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1), X_train, y_train, X_test, y_test)

    create_model_and_measure_rmse(AdaBoostRegressor(), X_train, y_train, X_test, y_test)

    # Classification models

    predictors = pd.DataFrame([data["carat"], data["x"], data["y"], data["z"]]).T


    #Encoding using one hot encoder

    predictors.head()
    one_hot_cut = pd.get_dummies(diamonds["cut"]).astype(float)
    one_hot_clarity = pd.get_dummies(diamonds["clarity"]).astype(float)
    predictors = predictors.join([one_hot_cut,one_hot_clarity])

    predictors = standard_scaler(predictors)

    target = data["color"]

    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20)

    # Create voting classifier for predicting the color

    voting_classifier_and_accuracy(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()