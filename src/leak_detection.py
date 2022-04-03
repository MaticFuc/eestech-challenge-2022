from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os
import numpy as np


class LeakDetection:
    """
    For correct automatic evaluation please implement your prediction logic inside this class
    """

    def __init__(self, dirname='.'):
        self.model = self.load_model(dirname)
        self.counter = 0
        self.count_all = 0
        self.predict_f = False
        self.prev = False
        self.counter_prev = 0
        self.batch = np.zeros((1,500))
        self.curr_batch = np.zeros((1,50))

    def load_model(self, dirname):
        """
        Loads your pretrained model to use it for prediction.
        Please use os.path.join(location_to_dir, model_file_name)

        :param dirname: Path to directory where model is located
        :return: your pretrained model, if no model is required return None

        Example:
            import os
            import joblib
            load(os.path.join(self.dirname, 'tree.joblib'))

        """
        with open(os.path.join(dirname, 'clf.pickle'),'rb') as f:
            clf = pickle.load(f)
        return clf

    def predict(self, features: List) -> bool:
        """
        Your implementation for prediction. If leak is detected it should return true.

        :param features: A list of features
        :return: should return true if leak is detected. Otherwise, it should return false.

        Example:
            return self.model.predict(features) == 0

        """
        X = np.array(features[1:])
        print(X)
        X[X == ''] = '0.0'
        X = X.reshape((1,len(X)))
        if self.predict_f and self.counter_prev >= 5:
            return True
        elif not self.predict_f:
            self.batch[0,5*self.counter:5*self.counter+5] = X
            self.counter = self.counter + 1
            if self.counter == 100:
                self.predict_f = True
            return self.prev
        else:
            
            self.batch = self.batch.reshape((1,500))
            self.batch = self.batch[0,5:]
            
            self.batch = np.append(self.batch,X.reshape((5,)),axis=0)
            self.batch = self.batch.reshape(1,-1)
        #print(X)
        #print(self.model.predict(X))
        for i in range(10):
            self.curr_batch[0,5*i:5*i+5] = self.batch[0,50*i:50*i+5]
        self.prev = self.model.predict(self.curr_batch)[0] == 0
        if self.prev:
            self.counter_prev = self.counter_prev + 1
        else:
            self.counter_prev = 0
        return self.model.predict(self.curr_batch)[0] == 0
        
