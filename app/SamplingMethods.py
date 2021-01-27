# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:48:17 2020

@author: Donovan
"""

def lowestPercentage(ml_model, X_test, n):
    """
    This sampling method finds the pictures with the lowest percent probability in the test set.
    It then removes the n samples with the lowest percent probability from X_test.
    Finaly it returns the new testing set and the sample set.
    
    Parameters
    ----------
    al_model: active learning class object
        The active learning model.
    n : int
        The number of samples to be returned.
        
    Returns
    -------
    X_sample: pandas DataFrame
        The new list of samples to be added to the train set.
    X_test: pandas DataFrame
        The new testing set with the samples removed.
    """
    from sklearn.utils import shuffle
    predictions, probabilities = ml_model.GetUnknownPredictions(X_test)

    X_test['prediction score'] = probabilities
    X_test.sort_values('prediction score', axis = 0)
    print(len(X_test.index.values))
    X_sample = X_test.iloc[:n, :]
    new_X_test = X_test.iloc[n:, :]
    print(len(X_sample.index.values))
    print(len(new_X_test.index.values))

    return list(shuffle(X_sample.index.values)), list(shuffle(new_X_test.index.values))