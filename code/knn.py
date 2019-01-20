"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        D = utils.euclidean_dist_squared(self.X, Xtest).transpose()
        D_ = np.argsort(D)
        DKNN_indices = D_[:,range(0,self.k)]
        
        t, k = DKNN_indices.shape
        y_ = np.zeros(t)
        for i in range(t):
            y_[i] = utils.mode(np.take(self.y, DKNN_indices[i])) 
                
        return y_