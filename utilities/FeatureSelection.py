
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

"""@package FeatureSelection
# ExtraTreesClassifier is an ensemble learning method fundamentally based on decision trees. 
# ExtraTreesClassifier, like RandomForest, randomizes certain decisions and subsets of data. 
"""
def extraTreesClassifier(fileIn):
    with open(fileIn, 'r') as f:
        data = pd.read_csv(fileIn)

    X = data.iloc[:,1:-1]  #feature columns
    y = data.iloc[:,-1]    #target column (last column)


    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.show()
    
extraTreesClassifier('csvOut.csv')