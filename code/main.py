import sys
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

import utils
import logReg
from logReg import logRegL2, kernelLogRegL2
from pca import PCA, AlternativePCA, RobustPCA

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == "1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # standard logistic regression
        lr = logRegL2(lammy=1)
        lr.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr.predict(Xtest) != ytest))

        utils.plotClassifier(lr, Xtrain, ytrain)
        utils.savefig("logReg.png")
        
        # kernel logistic regression with a linear kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_linear, lammy=1)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegLinearKernel.png")

    elif question == "1.1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # kernel logistic regression with a poly kernel
        poly_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_poly, lammy=0.01)
        poly_kernel.fit(Xtrain, ytrain)

        print("\nPOLY: Training error %.3f" % np.mean(poly_kernel.predict(Xtrain) != ytrain))
        print("POLY: Validation error %.3f" % np.mean(poly_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(poly_kernel, Xtrain, ytrain)
        utils.savefig("logRegPolynomialKernel.png")

        # kernel logistic regression with a RBF kernel
        RBF_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=0.01, sigma=0.5)
        RBF_kernel.fit(Xtrain, ytrain)

        print("\nRBF: Training error %.3f" % np.mean(RBF_kernel.predict(Xtrain) != ytrain))
        print("RBF: Validation error %.3f" % np.mean(RBF_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(RBF_kernel, Xtrain, ytrain)
        utils.savefig("logRegRBFKernel.png")


    elif question == "1.2":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        M1 = [-2, -1, 0, 1, 2]
        M2 = [-4, -3, -2, -1, 0]
        maxTrain = sys.maxsize
        maxTrainParams = (-1, -1)
        maxTest = sys.maxsize
        maxTestParams = (-1, -1)
        for i in range(len(M1)):
            for j in range(len(M2)):
                sigma = 10**M1[i]
                lammy = 10**M2[j]
                RBF_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=lammy, sigma=sigma)
                RBF_kernel.fit(Xtrain, ytrain)
                trainError = np.mean(RBF_kernel.predict(Xtrain) != ytrain)
                testError = np.mean(RBF_kernel.predict(Xtest) != ytest)
                if(trainError < maxTrain):
                    maxTrain = trainError
                    maxTrain_test = testError
                    maxTrainParams = (sigma, lammy)
                    maxTrainModel = RBF_kernel
                if(testError < maxTest):
                    maxTest = testError
                    maxTest_train = trainError
                    maxTestParams = (sigma, lammy)
                    maxTestModel = RBF_kernel
                
        
        print("\nRBF max: Training error %.3f" % maxTrain)
        print("RBF max: Validation error %.3f" % maxTrain_test)
        print("sigma: %.10f, lambda: %.10f" % (maxTrainParams[0], maxTrainParams[1]))
        print("\nRBF max: Validation error %.3f" % maxTest)
        print("RBF max: Training error %.3f" % maxTest_train)
        print("sigma: %.10f, lambda: %.10f" % (maxTestParams[0], maxTestParams[1]))

        utils.plotClassifier(maxTrainModel, Xtrain, ytrain)
        utils.savefig("logRegRBFKernel_MaxTrain.png")

        utils.plotClassifier(maxTestModel, Xtrain, ytrain)
        utils.savefig("logRegRBFKernel_MaxValidation.png")  

    elif question == '3':
        X = [
            [-4, 2], 
            [0,0],
            [-2, 1],
            [4, -2],
            [2, -1]
        ]
        miu = np.mean(X, axis = 0)
        X -= miu
        U,s,Vh= np.linalg.svd(X.T@X)

        x1 = [-3,2.5]
        x1_ =  x1 - miu
        x1Trans = x1@Vh[:1].T
        x1Loss = np.sum((x1Trans@Vh[:1]-x1_)**2)
        print(x1Loss)

        x2 = [-3,2]
        x2_ =  x2 - miu
        x2Trans = x2@Vh[:1].T
        x2Loss = np.sum((x2Trans@Vh[:1]-x2_)**2)
        print(x2Loss)
        
        trans = X@Vh[:1].T
        trans = np.insert(trans, 0, x1Trans[0])
        trans = np.insert(trans, 0, x2Trans[0])

        X = np.insert(X,0,x1, axis = 0)
        X = np.insert(X,0,x2, axis = 0)
        
        # Original 2d data
        plt.scatter(X[:,0], X[:,1])
        plt.show()
        # 1d transformation
        plt.scatter(trans, np.ones(len(X)))
        plt.show()

    elif question == '4.1': 
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        n,d = X.shape
        print(n,d)
        h,w = 64,64      # height and width of each image

        k = 5            # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(Xhat_pca[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X[i] - Xhat_pca[i])<threshold).reshape(h,w).T, cmap='gray', vmin=0, vmax=1)

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray', vmin=0, vmax=1)

            utils.savefig('highway_{:03d}.png'.format(i))

    else:
        print("Unknown question: %s" % question)    