import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
# inherits the predict() function from LeastSquares
class WeightedLeastSquares(LeastSquares): 
    def fit(self,X,y,z):
        # X^T z^T X w = X^T z^T y 
        self.w = solve(X.T@z.T@X, X.T@z.T@y)

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):
        # Calculate the function value
        f = np.sum(np.log(np.exp(X@w - y) + np.exp(y - X@w)))

        # Calculate the gradient value
        g = X.T@((np.exp(X@w - y) - np.exp(y - X@w))/(np.exp(X@w - y) + np.exp(y - X@w)))
        
        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        w_0 = np.ones(X.shape[0])
        X_ = np.c_[w_0,X]
        self.w = solve(X_.T@X_, X_.T@y)

    def predict(self, X):
        w_0 = np.ones(X.shape[0])
        X_ = np.c_[w_0,X]
        return X_@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        self.leastSquares.fit(self.__polyBasis(X), y)

    def predict(self, X):
        return self.leastSquares.predict(self.__polyBasis(X))

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        z = np.ones((X.shape[0],1))
        for i in range(self.p):
            for j in range(X.shape[1]):
                z = np.c_[z,X[:,j]**(i+1)]
        return z
