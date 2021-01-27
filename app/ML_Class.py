# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:54:45 2020
@author: Donovan
"""
import json
from json import JSONEncoder

class ML_Model:
    """
    This class creates a machine learning model based on the data sent,
    data preprocessing, and type of ml classifier.

    """

    def __init__(self, train_data, ml_classifier, preprocess):
        """
        This function controls the initial creation of the machine learning model.

        Parameters
        ----------
        train_data : pandas DataFrame
            The data the machine learning model will be built on.
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.

        Attributes
        -------
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
        X : pandas DataFrame
            The features in the train set.
        y : pandas Series
            The responce variable.
        ml_model : fitted machine learning classifier
            The machine learning model created using the training data.
        """
        self.ml_classifier = ml_classifier
        self.preprocess = preprocess

        self.X = train_data.iloc[:,: -1].values
        self.y = train_data.iloc[:, -1].values

        self.X = self.preprocess.fit_transform(self.X)

        self.ml_model = ml_classifier.fit(self.X, self.y)

    def GetKnownPredictions(self, new_data):
        """
        This function predicts the labels for a new set of data that contains labels.
        It returns these predictions and the probability.

        Parameters
        ----------
        new_data : pandas DataFrame
            The new data to be labeled.

        Returns
        -------
        y_prediction : list
            list of predicted labels.
        prob : list
            The probability that the label is correct.
        """
        new_data_X = new_data.iloc[:, :-1].values
        new_data_X = self.preprocess.transform(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        y_probabilities = self.ml_model.predict_proba(new_data_X)
        y_probabilities = [max(prob) for prob in y_probabilities]
        return y_prediction, max(y_probabilities)

    def GetUnknownPredictions(self, new_data_X):
        """
        This function predicts the labels for a new set of data that does not contains labels.
        It returns these predictions and the probability.

        Parameters
        ----------
        new_data : pandas DataFrame
            The new data to be labeled.

        Returns
        -------
        y_prediction : list
            list of predicted labels.
        prob : list
            The probability that the label is correct.
        """
        new_data_X = self.preprocess.transform(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        y_probabilities = self.ml_model.predict_proba(new_data_X)
        y_probabilities = [max(prob) for prob in y_probabilities]
        return y_prediction, y_probabilities

    def K_fold(self):
        """
        This function performs a 10-fold cross-validation and returns the accuracies of each fold.

        Returns
        -------
        accuracies : list
            The 10 accuracy values using 10-fold cross-validation.
        """
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(self.ml_classifier, self.X, self.y, cv=3)
        return accuracies

    def infoForProgress(self, train_img_names):
        """
        This function returns the information nessessary to display the progress of the active learning model.

        Parameters
        ----------
        train_img_names : list
            list of image names in the training set.

        Returns
        -------
        health_pic : list
            List of images that were predicted as healthy.

        blight_pic : list
            List of images that were predicted as unhealthy.
        """
        y_actual = self.y
        y_pic = train_img_names
        #y_pred = self.ml_model.predict(self.X)
        #y_pred = list(y_pred)
        health_pic = []
        blight_pic = []
        if len(y_pic) == 10:
            y_pic = y_pic[::-1]
            for y_idx, y in enumerate(y_actual):
                if y == 'H':
                    health_pic.append(y_pic[y_idx])
                elif y == 'B':
                    blight_pic.append(y_pic[y_idx])
        else:
            y_pic_head = y_pic[:10]
            y_pic_head_rev = y_pic_head[::-1]
            y_pic_result = y_pic_head_rev
            y_pic_tail = y_pic[10:]
            y_pic_tail_length = len(y_pic_tail)
            for i in range(5,y_pic_tail_length + 5,5):
                y_pic_tail_rev = y_pic_tail[:5]
                y_pic_tail_rev = y_pic_tail_rev[::-1]
                y_pic_result = y_pic_result + y_pic_tail_rev
                y_pic_tail = y_pic_tail[5:]
            for y_idx, y in enumerate(y_actual):
                if y == 'H':
                    health_pic.append(y_pic_result[y_idx])
                elif y == 'B':
                    blight_pic.append(y_pic_result[y_idx])
        return health_pic, blight_pic

    def infoForResults(self, train_img_names, test):
        """
        This function returns the information nessessary to display the final results of the active learning model.

        Parameters
        ----------
        train_img_names : list
            list of image names in the training set.
        test : pandas dataframe
            The test set of the machine learning model.

        Returns
        -------
        health_pic_user : list
            List of images that were predicted as healthy.

        blight_pic_user : list
            List of images that were predicted as blight.

        health_pic : list
            List of images in the test set that are predicted to being healthy.

        blight_pic : list
            List of images in the test set that are predicted to being blighted.
        """
        health_pic_user, blight_pic_user = self.infoForProgress(train_img_names)
        test_pic = list(test.index.values)
        y_pred, y_prob = self.GetUnknownPredictions(test)
        health_pic = []
        blight_pic = []
        health_pic_prob = []
        blight_pic_prob = []
        for y_idx, y in enumerate(y_pred):
            if y == 'H':
                health_pic.append(test_pic[y_idx])
                health_pic_prob.append(y_prob[y_idx])
            elif y == 'B':
                blight_pic.append(test_pic[y_idx])
                blight_pic_prob.append(y_prob[y_idx])
        health_list = list(zip(health_pic,health_pic_prob))
        blight_list = list(zip(blight_pic,blight_pic_prob))
        health_list_sorted = sorted(health_list, reverse=True, key = lambda x: x[1])
        blight_list_sorted = sorted(blight_list, reverse=True, key = lambda x: x[1])
        if health_pic and health_pic_prob:
            new_health_pic, new_health_pic_prob = list(zip(*health_list_sorted))
        else:
            new_health_pic = []
            new_health_pic_prob = []
        if blight_pic and blight_pic_prob:
            new_blight_pic, new_blight_pic_prob = list(zip(*blight_list_sorted))
        else:
            new_blight_pic = []
            new_blight_pic_prob = []

        return health_pic_user, blight_pic_user, new_health_pic, new_blight_pic, new_health_pic_prob, new_blight_pic_prob

class Active_ML_Model:
    """
    This class creates an active learning model based on the data sent,
    data preprocessing, and type of ml classifier.

    """
    def __init__(self, data, ml_classifier, preprocess, n_samples = 10):
        """
        This function controls the initial creation of the active learning model.

        Parameters
        ----------
        data : pandas DataFrame
            The data the active learning model will be built on.
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
        n_samples : int
            The number of random samples to be used in the initial model creation.

        Attributes
        -------
        ml_classifier : classifier object
            The classifier to be used to create the active learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
        test : pandas DataFrame
            The training set.
        train : pandas DataFrame
            The train set.
        """
        from sklearn.utils import shuffle
        data = shuffle(data)
        self.sample = data.iloc[:n_samples, :]
        self.test = data.iloc[n_samples:, :]
        self.train = None
        self.ml_classifier = ml_classifier
        self.preprocess = preprocess

    def Train(self, sample):
        """
        This function trains the innitial ml_model
        Parameters
        ----------
        train : pandas DataFrame
            The training set with labels

        Attributes Added
        ----------------
        ml_model : fitted machine learning classifier
            The machine learning model created using the training data.
        """
        import pandas as pd
        if self.train != None:
            self.train = pd.concat([self.train, sample])
        else:
            self.train = sample
        self.ml_model = ML_Model(self.train, self.ml_classifier, self.preprocess)

    def Continue(self, sampling_method, n_samples = 5):
        """
        This function continues the active learning model to the next step.

        Parameters
        ----------
        sampling_method : Python Function
            Determines the next set of samples to send to user.
        n_samples : int
            The number of samplest that should be added the the train set.

        Attributes Updated
        -------
        ml_classifier : classifier object
            The classifier to be used to create the active learning model.
        test : pandas DataFrame
            The training set.
        train : pandas DataFrame
            The train set.
        """
        import pandas as pd
        self.sample, self.test = sampling_method(self.ml_model, n_samples)

    def infoForProgress(self):
        """
        This function returns the information nessessary to display the progress of the active learning model.

        Returns
        -------
        health_pic : list
            List of images that were predicted as healthy.

        blight_pic : list
            List of images that were predicted as unhealthy.
        """
        y_actual = self.ml_model.train['y_value']
        y_pic = list(self.ml_model.train.index)
        #y_pred, y_prob = self.ml_model.GetKnownPredictions(self.ml_model.train)
        #y_pred = list(y_pred)
        health_pic = []
        blight_pic = []
        if len(y_pic) == 10:
            y_pic = y_pic[::-1]
            for y_idx, y in enumerate(y_actual):
                if y == 'H':
                    health_pic.append(y_pic[y_idx])
                elif y == 'B':
                    blight_pic.append(y_pic[y_idx])
        else:
            y_pic_head = y_pic[:10]
            y_pic_head_rev = y_pic_head[::-1]
            y_pic_result = y_pic_head_rev
            y_pic_tail = y_pic[10:]
            y_pic_tail_length = len(y_pic_tail)
            for i in range(5,y_pic_tail_length + 5,5):
                y_pic_tail_rev = y_pic_tail[:5]
                y_pic_tail_rev = y_pic_tail_rev[::-1]
                y_pic_result = y_pic_result + y_pic_tail_rev
                y_pic_tail = y_pic_tail[5:]
            for y_idx, y in enumerate(y_actual):
                if y == 'H':
                    health_pic.append(y_pic_result[y_idx])
                elif y == 'B':
                    blight_pic.append(y_pic_result[y_idx])
        return health_pic, blight_pic

    def infoForResults(self):
        """
        This function returns the information nessessary to display the final results of the active learning model.

        Returns
        -------
        health_pic_user : list
            List of images that were predicted correctly.

        blight_pic_user : list
            List of images that were predicted incorrectly.

        health_pic : list
            List of images in the test set that are predicted to being healthy.

        blight_pic : list
            List of images in the test set that are predicted to being blighted.
        """
        health_pic_user, blight_pic_user = self.infoForProgress()
        test_pic = list(self.ml_model.train.idx)
        y_pred, y_prob = self.ml_model.GetUnknownPredictions(self.ml_model.test)
        health_pic = []
        blight_pic = []
        health_pic_prob = []
        blight_pic_prob = []
        for y_idx, y in enumerate(y_pred):
            if y == 'H':
                health_pic.append(test_pic[y_idx])
                health_pic_prob.append(y_prob[y_idx])
            elif y == 'B':
                blight_pic.append(test_pic[y_idx])
                blight_pic_prob.append(y_prob[y_idx])
        health_list = list(zip(health_pic,health_pic_prob))
        blight_list = list(zip(blight_pic,blight_pic_prob))
        health_list_sorted = sorted(health_list, reverse=True, key = lambda x: x[1])
        blight_list_sorted = sorted(blight_list, reverse=True, key = lambda x: x[1])
        new_health_pic, new_health_pic_prob = list(zip(*health_list_sorted))
        new_blight_pic, new_blight_pic_prob = list(zip(*blight_list_sorted))
        if health_pic and health_pic_prob:
            new_health_pic, new_health_pic_prob = list(zip(*health_list_sorted))
        else:
            new_health_pic = []
            new_health_pic_prob = []
        if blight_pic and blight_pic_prob:
            new_blight_pic, new_blight_pic_prob = list(zip(*blight_list_sorted))
        else:
            new_blight_pic = []
            new_blight_pic_prob = []

        return health_pic_user, blight_pic_user, new_health_pic, new_blight_pic, new_health_pic_prob, new_blight_pic_prob

class AL_Encoder(JSONEncoder):
    """
    This class attempts to make the Active_ML_Model JSON serializable.

    Warning
    -------
    Active_ML_Model is not JSON serializable and thus this class does not work.
    """
    def default(self, o):
        return o.__dict__