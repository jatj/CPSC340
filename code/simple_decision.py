# standard Python imports
import os
import pickle
# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
# CPSC 340 code
import utils

class DecisionStumpHardCoded:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        # Info learned from the depth 2 decision three using DecisionStumpInfoGain
        splitVariables = [0,1,1]
        splitVals = [-81.0,40.0,39.0]
        splitSats = [0,0,0]
        splitNots = [1,0,1]

        M, D = X.shape

        yhat = np.zeros(M)

        for m in range(M):
            # We check for the split values in a hierarchical order
            if X[m, splitVariables[0]] > splitVals[0]:
                if X[m, splitVariables[1]] > splitVals[1]:
                    yhat[m] = splitSats[1]
                else:
                    yhat[m] = splitNots[1]
            else:
                if X[m, splitVariables[2]] > splitVals[2]:
                    yhat[m] = splitSats[2]
                else:
                    yhat[m] = splitNots[2]

        return yhat

if __name__ == "__main__":
    # Get dataset
    with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    X = dataset["X"]
    y = dataset["y"]

    # Predict with the model
    model = DecisionStumpHardCoded()
    y_pred = model.predict(X)

    # Calculate the error
    error = np.mean(y_pred != y)
    print("Error: %.3f" % error)

    # Plot boundary regions
    utils.plotClassifier(model, X, y)

    fname = os.path.join("..", "figs", "q6_4_1_HardCodedDecisionBoundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)