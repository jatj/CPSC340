import argparse
import numpy as np
import sys

import utils
import linear_model

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logReg(maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.1":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL2(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL1(L1_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.5":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # TODO
        logRegL2 = LogisticRegression('l2', C=1, solver='liblinear', fit_intercept=False)
        logRegL2.fit(XBin, yBin)
        logRegL2TrainErr = utils.classification_error(logRegL2.predict(XBin),yBin)
        logRegL2ValidErr = utils.classification_error(logRegL2.predict(XBinValid),yBinValid)
        
        print("Scikit L2")
        print("\nTraining error %.3f" % utils.classification_error(logRegL2.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(logRegL2.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (logRegL2.coef_ != 0).sum())

        logRegL1 = LogisticRegression('l1', C=1, solver='liblinear', fit_intercept=False)
        logRegL1.fit(XBin, yBin)
        logRegL1TrainErr = utils.classification_error(logRegL1.predict(XBin),yBin)
        logRegL1ValidErr = utils.classification_error(logRegL1.predict(XBinValid),yBinValid)

        print("\n\nScikit L1")
        print("\nTraining error %.3f" % utils.classification_error(logRegL1.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(logRegL1.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (logRegL1.coef_ != 0).sum())

    elif question == "3":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        print(np.unique(model.predict(XMulti)))


    elif question == "3.2":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))
        print("# nonZeros: %d" % (model.W != 0).sum())

    elif question == "3.4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))
        print("# nonZeros: %d" % (model.W != 0).sum())

    elif question == "3.5":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # TODO

        oneVSall = LogisticRegression(C=9999, multi_class='ovr', solver='liblinear', fit_intercept=False)
        oneVSall.fit(XMulti, yMulti)
        oneVSallTrainErr = utils.classification_error(oneVSall.predict(XMulti),yMulti)
        oneVSallValidErr = utils.classification_error(oneVSall.predict(XMultiValid),yMultiValid)

        print("\nScikit Multiclass One vs All")
        print("\nTraining error %.3f" % utils.classification_error(oneVSall.predict(XMulti),yMulti))
        print("Validation error %.3f" % utils.classification_error(oneVSall.predict(XMultiValid), yMultiValid))
        print("# nonZeros: %d" % (oneVSall.coef_ != 0).sum())


        softmaxModel = LogisticRegression(C=9999, multi_class='multinomial', solver='lbfgs', fit_intercept=False)
        softmaxModel.fit(XMulti, yMulti)
        softmaxModelTrainErr = utils.classification_error(softmaxModel.predict(XMulti),yMulti)
        softmaxModelValidErr = utils.classification_error(softmaxModel.predict(XMultiValid),yMultiValid)

        print("\nScikit Multiclass SoftMax")
        print("\nTraining error %.3f" % utils.classification_error(softmaxModel.predict(XMulti),yMulti))
        print("Validation error %.3f" % utils.classification_error(softmaxModel.predict(XMultiValid), yMultiValid))
        print("# nonZeros: %d" % (softmaxModel.coef_ != 0).sum())