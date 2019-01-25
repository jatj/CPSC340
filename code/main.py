# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        testErrors = np.zeros(15)
        trainingErrors = np.zeros(15)
        depths = range(1,16)
        for d in depths:
            model = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=1)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)

            testErrors[d-1] = te_error
            trainingErrors[d-1] = tr_error

        plt.plot(depths, trainingErrors, label="Training")
        plt.plot(depths, testErrors, label="Testing", linestyle=":", linewidth=3)
        plt.xlabel("Depth of tree")
        plt.ylabel("Error")
        plt.legend()
        fname = os.path.join("..", "figs", "q1_1_errors.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape
        half_n = int(n/2)
        Xs = [X[0:half_n],X[half_n:]]
        Ys = [y[0:half_n],y[half_n:]]

        for i in range(2):
            Xtrain = Xs[i]
            ytrain = Ys[i]

            Xvalidation = Xs[(i+1)%2]
            yvalidation = Ys[(i+1)%2]

            validationErrors = np.zeros(15)
            trainingErrors = np.zeros(15)
            depths = range(1,16)
            for d in depths:
                model = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=1)
                model.fit(Xtrain, ytrain)

                y_pred = model.predict(Xtrain)
                tr_error = np.mean(y_pred != ytrain)

                y_pred = model.predict(Xvalidation)
                va_error = np.mean(y_pred != yvalidation)

                validationErrors[d-1] = va_error
                trainingErrors[d-1] = tr_error

            plt.plot(depths, trainingErrors, label="Training")
            plt.plot(depths, validationErrors, label="Validation", linestyle=":", linewidth=3)

            diffError = validationErrors - trainingErrors
            aveError = (validationErrors + trainingErrors)/2
            stats_rows = ["gap","validation", "train", "average"]
            stats_cols = range(1,16)
            stats = pd.DataFrame(index=stats_rows, columns=stats_cols, data = [
                diffError,
                validationErrors,
                trainingErrors,
                aveError
            ])
            print(stats)

            plt.xlabel("Depth of tree")
            plt.ylabel("Error")
            plt.title("Error over depth")
            plt.legend()
            fname = os.path.join("..", "figs", "q1_2_training_validation_{}.pdf".format(i+1))
            plt.savefig(fname)
            plt.clf()
            print("\nFigure saved as '%s'" % fname)



    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        # ############################ 2.2.1
        print("\n############################ 2.2.1\n")
        print("The word of column 51: {}".format(wordlist[50]))

        # ############################ 2.2.2
        print("\n############################ 2.2.2\n")
        print("Words present in training example 501:")
        print(wordlist[X[500] > 0])

        # ############################ 2.2.3
        print("\n############################ 2.2.3\n")
        print("Newsgroup name of training example 501:")
        print(groupnames[y[500]])

    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        # Scikit learn BernoulliNB
        sciKit_naive_bayes = BernoulliNB()
        sciKit_naive_bayes.fit(X,y)
        y_pred = sciKit_naive_bayes.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (scikit) validation error: %.3f" % v_error)
    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']
        knnModels = []
        knnScikitModels = []

        for k in [1,3,10]:
            print("\n------ K={} -----\n".format(k))
            
            modelKNN = KNN(k)
            modelKNN.fit(X,y)
            y_pred = modelKNN.predict(X)
            training_error = np.mean(y_pred != y)
            y_pred = modelKNN.predict(Xtest)
            test_error = np.mean(y_pred != ytest)
            print("KNN (k={:d}) training_error: {:.3f}".format(k, training_error))
            print("KNN (k={:d}) test_error: {:.3f}".format(k,test_error))

            modelKNN_Scikit = KNeighborsClassifier(n_neighbors=k)
            modelKNN_Scikit.fit(X,y)
            y_pred = modelKNN_Scikit.predict(X)
            training_error = np.mean(y_pred != y)
            y_pred = modelKNN_Scikit.predict(Xtest)
            test_error = np.mean(y_pred != ytest)
            print("KNN Scikit (k={:d}) training_error: {:.3f}".format(k, training_error))
            print("KNN Scikit (k={:d}) test_error: {:.3f}".format(k, test_error))

            knnModels.append(modelKNN)
            knnScikitModels.append(modelKNN_Scikit)

        utils.plotClassifier(knnModels[0],X,y)
        fname = os.path.join("..", "figs", "q3_3_myKNN.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        plt.clf()

        utils.plotClassifier(knnScikitModels[0],X,y)
        fname = os.path.join("..", "figs", "q3_3_scikitKNN.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("Random tree")
        evaluate_model(RandomTree(np.inf))
        print("Random forest")
        evaluate_model(RandomForest(np.inf,50))
        print("Random forest scikit")
        evaluate_model(RandomForestClassifier(n_estimators=50))



    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        lowestError = np.inf
        lowestKMeans = None
        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            error = model.error(X)
            if error < lowestError:
                lowestError = error
                lowestKMeans = model
    
        print("KMeans lowest error: {}".format(lowestError))
        y = lowestKMeans.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "q_5_1_kmeans.pdf")
        plt.title("Cluster Plot")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']
        lowestErrors = []
        ks = range(1,11)
        for k in ks:
            lowestError = np.inf
            for i in range(50):
                model = Kmeans(k=k)
                model.fit(X)
                error = model.error(X)
                if error < lowestError:
                    lowestError = error
                    lowestKMeans = model
            lowestErrors.append(lowestError)


        plt.plot(ks, lowestErrors, label="Minimum errors")
        plt.xlabel("K")
        plt.ylabel("Error")
        plt.legend()
        plt.title("Kmeans errors over k")
        fname = os.path.join("..", "figs", "q5_2_Kmeans_errors.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=1, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
