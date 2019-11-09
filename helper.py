from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def encoder(var):
    """
    Encodes data from strings to numbers
    :return:
    """
    encoder = LabelEncoder()
    enc_var = encoder.fit_transform(var)
    var = enc_var

    return var

def correlation(data):
    plt.figure(figsize=(6, 6))
    correlation = data.corr()
    return sns.heatmap(correlation,cmap='YlGnBu',vmax=1, annot=True,square=True)

def standard_scaler(data):
    std_scaler = preprocessing.StandardScaler()
    data = std_scaler.fit_transform(data)
    return data

def create_model_and_measure_rmse(model, X_train, y_train, X_test, y_test):
    """
   Creates a model and trains it.
    :return:
    """
    print("Starting training the model {}...".format(model))
    model.fit(X_train,y_train)
    print("Done!")
    print("Using the trained model for predicting the test set..")
    predictions = model.predict(X_test)
    print("Done!")
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return print("The root mean squared error for {} is:".format(model), rmse)

def voting_classifier_and_accuracy(X_train, y_train, X_test, y_test):
    """
    Create a voting classifier from Logistic Regressionn Random Forest, SVM, Decision tree, trains it and use it to predict test data.
    :return:
    """

    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC()
    dt_clf=DecisionTreeClassifier()

    voting_clf = VotingClassifier(estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf),("dt",dt_clf)], voting='hard')
    voting_clf.fit(X_train,y_train)


    for clf in (log_clf, rnd_clf, svm_clf, dt_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("The accuracy for: ",clf.__class__.__name__, "is: ",accuracy_score(y_test, y_pred))