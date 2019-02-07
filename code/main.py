
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

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        # YOUR CODE HERE FOR Q1.1.1
        print("\n##################### Q1.1.1\n")
        sumStars = np.sum(X, axis=0)
        maxStarsItemIndex = sumStars.argmax(axis=1)[0,0]
        maxStarsItemID = item_inverse_mapper[maxStarsItemIndex]
        maxStars = sumStars[0,maxStarsItemIndex]
        print("Max star product index: {}, id: {}, stars: {}, url: https://www.amazon.com/dp/{}".format(maxStarsItemIndex, maxStarsItemID, maxStars, maxStarsItemID))

        # YOUR CODE HERE FOR Q1.1.2
        print("\n##################### Q1.1.2\n")
        sumRatings = np.sum(X_binary, axis=1)
        maxRatingsUserIndex = sumRatings.argmax(axis=0)[0,0]
        maxRatingsUserID = user_inverse_mapper[maxRatingsUserIndex]
        maxRatings = sumRatings[maxRatingsUserIndex,0]
        print("User with max ratings index: {}, id: {}, # ratings: {}".format(maxRatingsUserIndex, maxRatingsUserID, maxRatings))
        
        # YOUR CODE HERE FOR Q1.1.3
        print("\n##################### Q1.1.3.1\n")
        userRatings = X_binary.getnnz(axis=1)
        n_bins = maxRatings
        plt.hist(userRatings, bins=n_bins)
        plt.yscale("log", nonposy="clip")
        plt.xlabel('# of ratings')
        plt.ylabel('# of users')
        plt.title("Ratings per user 1.1.3.1")
        fname = os.path.join("..", "figs", "ratings_per_user.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        plt.clf()
        print("\n##################### Q1.1.3.2\n")
        itemRatings = X_binary.getnnz(axis=0)
        n_bins = np.max(itemRatings, axis=0)
        plt.hist(itemRatings, bins=n_bins)
        plt.yscale("log", nonposy="clip")
        plt.xlabel('# of ratings')
        plt.ylabel('# of items')
        plt.title("Ratings per item 1.1.3.2")
        fname = os.path.join("..", "figs", "ratings_per_item.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        plt.clf()
        print("\n##################### Q1.1.3.3\n")
        allRatings = X.data
        plt.hist(allRatings, bins=5)
        plt.xlabel('# of stars')
        plt.ylabel('# of ratings')
        plt.title("Ratings 1.1.3.3")
        fname = os.path.join("..", "figs", "ratings.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        plt.clf()

    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2

        # Take out grill_brush from the data 
        cols = np.arange(X.shape[1])
        filterGrill = np.where(np.logical_not(np.in1d(cols, [grill_brush_ind])))[0]
        X_filtered = X[:, filterGrill]
        # Transpose data for KNN on items (columns)
        grill_brush_vec_T = grill_brush_vec.transpose()
        X_filtered_T = X_filtered.transpose()
        X_binary_filtered_T = X_filtered_T != 0

        # Wrapper function to call KNN on X data  with the provided metric and find KNN of item
        def doKNN(X,item,metric):
            knn = NearestNeighbors(metric=metric)
            knn.fit(X)
            knDistances, knn = knn.kneighbors(item)
            knDistances = knDistances[0,:]
            knnIndices = knn[0,:]
            knnIDs = []
            knnLinks = []
            for i in range(knn.shape[1]):
                knnIDs.append(item_inverse_mapper[knnIndices[i]])
                knnLinks.append(url_amazon % knnIDs[i])
            print(knnIDs)
            print(knDistances)
            print(knnLinks)
            return knnIndices, knnIDs

        print("\n##################### Q1.2.1\n")
        knnEuclideanNeighbors, knnEuclideanNeighborsIDs = doKNN(X_filtered_T,grill_brush_vec_T,"euclidean")

        print("\n##################### Q1.2.2\n")
        knnNormalizedEuclideanNeighbors, knnNormalizedEuclideanNeighborsIDs = doKNN(normalize(X_filtered_T),grill_brush_vec_T,"euclidean")

        print("\n##################### Q1.2.3\n")
        knnCosineNeighbors, knnCosineNeighborsIDs = doKNN(X_filtered_T,grill_brush_vec_T,"cosine")


        # YOUR CODE HERE FOR Q1.3

        print("\n##################### Q1.3\n")
        # Euclidean reviews
        cols = np.arange(X.shape[1])
        X_knnEuclideanSum = X_binary[:, knnEuclideanNeighbors].sum(axis=0)
        print("Euclidean reviews:")
        print(knnEuclideanNeighborsIDs)
        print(X_knnEuclideanSum)

        # Cosine reviews
        cols = np.arange(X.shape[1])
        X_knnCosineSum = X_binary[:, knnCosineNeighbors].sum(axis=0)
        print("\Cosine reviews:")
        print(knnCosineNeighborsIDs)
        print(X_knnCosineSum)


    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']
        V = np.diag(np.append(np.ones(400), np.full(100, 0.1)))

        # Fit weighted least-squares estimator
        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,V)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Weighted Least Squares",filename="weighted_least_squares_outliers.pdf")

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares robust estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares with bias model
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, bias",filename="least_squares_bias.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        eTrain = []
        eTest = []

        for p in range(11):
            print("p=%d" % p)

            # Fit least-squares poly model
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            eTrain.append(np.mean((model.predict(X) - y)**2))
            eTest.append(np.mean((model.predict(Xtest) - ytest)**2))

        plt.plot(range(11), eTrain, label="Training")
        plt.plot(range(11), eTest, label="Testing")
        plt.xlabel("Polynomial degree")
        plt.ylabel("Error")
        plt.title("Least squares, poly")
        plt.legend()
        fname = os.path.join("..", "figs", "least_squares_poly.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        print("\nTraining errors")
        print(eTrain)
        print("\nTraining errors")
        print(eTest)

    else:
        print("Unknown question: %s" % question)

