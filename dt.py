# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from dtreeviz.trees import *

# Function importing Dataset
def importdata():
    balance_data = pd.read_csv('diabetes.csv', sep=',', header=None)

    # One-hot encode the categorical features (assuming columns 1 and 2 are categorical)
    balance_data = pd.get_dummies(balance_data, columns=[1, 2])
    # One-hot encode the categorical features (assuming columns 1 and 2 are categorical)
    balance_data.columns = balance_data.columns.astype(str)


    # Printing the dataset shape
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)

    # Printing the dataset observations
    print("Dataset:", balance_data.head())
    return balance_data

# Function to split the dataset
def splitdataset(balance_data):
    X = balance_data.drop('Outcome', axis=1)  # Assuming 'Outcome' is the name of your target variable
    Y = balance_data['Outcome']  # Assuming 'Outcome' is the name of your target variable

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex
def train_using_gini(X_train, X_test, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to make predictions
def prediction(X_test, clf_object):

    # Prediction on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

    print("Report : ", classification_report(y_test, y_pred))

# Function to visualize the decision tree using dtreeviz
def visualize_decision_tree(clf, feature_names, target_name, class_names):
    viz = dtreeviz(clf, feature_names=feature_names, target_name="Outcome", class_names=["No Diabetes", "Diabetes"])
    viz.view()

# Driver code
def main():
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)

    # Operational Phase
    print("Results Using Gini Index")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    # Visualize the decision tree
    feature_names = X.columns
    target_name = "Outcome"  # Adjust based on your dataset
    class_names = [str(x) for x in clf_gini.classes_]
    visualize_decision_tree(clf_gini, feature_names, target_name, class_names)

# Calling main function
if __name__ == "__main__":
    main()
