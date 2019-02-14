
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils
from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree

def bezdekIris_dataset(filename = "bezdekIris.data.txt", classesToCompare = (b'Iris-setosa',b'Iris-versicolor')):
    names = ("sepal length", "sepal width", "petal length", "petal width", "class")
    classes = (b'Iris-setosa',b'Iris-versicolor',b'Iris-virginica')
    formats = ("f","f","f","f","S15")
    data = np.loadtxt(os.path.join('..','data',filename), delimiter=',', dtype={'names': names, 'formats': formats})
    
    X = np.zeros((data.shape[0],2))
    for i in range(2):
        X[:,i] = data[names[i]]
    
    outputRanges = []
    y = np.zeros(data.shape[0], dtype="int")
    classIndex = 0
    for i in range(len(classes)):
        if classes[i] in classesToCompare:
            y[np.isin(data["class"],classes[i])] = classIndex
            classIndex += 1
            outputRanges = outputRanges + list(range(i*50, i*50+50))

    return (X[outputRanges,:], y[outputRanges], names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--excercise', required=True)
    io_args = parser.parse_args()
    excercise = io_args.excercise

    # PERCEPTRON
    if excercise == "1":
        (X,y,names) = bezdekIris_dataset(classesToCompare=(b'Iris-virginica',b'Iris-versicolor'))
        print(y)
        # Fit least-squares model
        model = DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain)
        model.fit(X,y)
        utils.plotClassifier(model,X,y)
        plt.show()

    else:
        print("Unknown excercise: %s" % excercise)

