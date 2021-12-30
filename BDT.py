import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  AdaBoostClassifier
import joblib
from utils.draw_results import prepPredArr, getROCaxes, drawOut, draw_ROC
np.random.seed(1)

def BDTmodel(X, Y, X_test):
    """ training Particle identification Discriminator use BDT method
        Input paras: X(training samples variables); Y(training samples labels); X_test(testing samples variables)
    return:
        Predict labels for training and testing sample
    """
    base_dt = tree.DecisionTreeClassifier(max_depth=5)
    clf=AdaBoostClassifier(base_estimator = base_dt, learning_rate=1,n_estimators=10)
    clf = clf.fit(X,Y)

    Ypred_test = clf.predict_proba(X_test)[:,1:]
    Ypred_train = clf.predict_proba(X)[:,1:]
    joblib.dump(clf, "./model/BDT_train_model.m")

    params = clf.get_params(deep=True)
    print("params",params)
    print("feature_importances", clf.feature_importances_)
    print("list of classifiers", clf.estimators_)
    print("estimator weights", clf.estimator_weights_)
    print("n features in", clf.n_features_in_)
    print("number of classes", clf.n_classes_)

    return Ypred_test, Ypred_train

def main():
    """
    Training model, do overfit test and draw ROC curve
    """
    data = np.load("./train_data/EXO-200_training-All-Phase1-data-BDT-1797602evt-600KeVcut_nonorm_3D0.6.npz")
    X = data['report']
    Y = data['label']

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=3)
    Ypred_test, Ypred_train = BDTmodel(Xtrain, Ytrain, Xtest)
    print(len(Ypred_train))
    YpredSgts,YpredSgtr,YpredBgts,YpredBgtr = prepPredArr(Ytrain, Ytest,Ypred_train, Ypred_test)
    drawOut(YpredSgts, YpredSgtr, YpredBgts, YpredBgtr)
    nSgEffTr,nSgEffTs,nBgRejTr,nBgRejTs = getROCaxes(YpredSgts,YpredSgtr, YpredBgts,YpredBgtr)
    draw_ROC(nSgEffTr,nSgEffTs,nBgRejTr,nBgRejTs)

if __name__ == '__main__':
    main()
