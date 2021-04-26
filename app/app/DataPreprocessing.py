# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:21:32 2020

@author: Donovan
"""
class DataPreprocessing:
    """
    This class object prepares the data for classification.
    
    """
    
    def __init__(self, standard_scaling = False, normalization = False, pca = False, components = None):
        """
        This function controls the initial creation of the data preprocessing class object.
        
        Parameters
        ----------
        standard_scaling : Boolean
            If true use standard scaling
        normalization : Boolean
            If true use normalization
        pca : Boolean
            If true use principal component analysis
        components : int
            The number of components PCA should have
            
        Attributes
        -------
        sc : standard scaler object or None
            Used to scale features
        norm : Normalizer object or None
            Used to normalize features
        pca : PCA object
            Used to perform principal component analysis on features
        """
        self.sc = None
        if standard_scaling == True:
            from sklearn.preprocessing import StandardScaler
            self.sc = StandardScaler()
        
        self.norm = None
        if normalization == True:
            from sklearn.preprocessing import Normalizer
            self.norm = Normalizer()
        
        self.pca = None
        if pca == True:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components = components)

        
    def fit_transform(self, X_train):
        """
        This function fits and transforms the data.
        
        Parameters
        ----------
        X_train : pandas DataFrame
            The data to be transformed.
            
        Attributes Modified
        -------------------
        sc : standard scaler object or None
            Used to scale features fitted to the data
        norm : Normalizer object or None
            Used to normalize features fitted to the data
        pca : PCA object
            Used to perform principal component analysis on features fitted to the data
        
        Returns
        -------
        X_train : pandas DataFrame
            The preprocessed data.
        """
        if self.sc != None:
            X_train = self.sc.fit_transform(X_train)
        if self.norm != None:
            X_train = self.norm.fit_transform(X_train)
        if self.pca != None:
            X_train = self.pca.fit_transform(X_train)
        return X_train
    
    def transform(self, X_test):
        """
        Transforms new data before prediction.
        
        Parameters
        ----------
        X_test : pandas DataFrame
            The data to be transformed.
        
        Returns
        -------
        X_test : pandas DataFrame
            The preprocessed data.
        """

        if self.sc != None:
            X_test = self.sc.transform(X_test)
        if self.norm != None:
            X_test = self.norm.transform(X_test)
        if self.pca != None:
            X_test = self.pca.transform(X_test)
        return X_test