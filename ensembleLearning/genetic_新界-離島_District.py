from Frame import Genetic_geatpy # 导入库
from sklearn import datasets     # 波士頓房價數據集
import os
import pandas as pd
from sklearn import preprocessing
import numpy as np

def myTrain_test_split(path, splitYear=2018):
    data = pd.read_csv(path)
    testData = data[data["Y3_year"] >= splitYear]
    trainData = data[data["Y3_year"] < splitYear]
    print(trainData.shape, testData.shape)


    Data = pd.concat([trainData, testData])
    print(Data["Y3_year"].head(200))

    npData = Data.values

    X = npData[:, :len(npData[0]) - 1]
    y = npData[:, len(npData[0]) - 1:]

    zscore = preprocessing.StandardScaler()
    X_zscore = zscore.fit_transform(X)

    train_X = X_zscore[:trainData.shape[0]]
    test_X = X_zscore[trainData.shape[0]:]
    train_y = y[:trainData.shape[0]]
    test_y = y[trainData.shape[0]:]
    print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)
    return train_X, test_X, train_y, test_y


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = myTrain_test_split("../dataset/BayesianRidge_economics_finished_新界-離島.csv", splitYear=2018)
    Y_train, Y_test = Y_train.reshape(-1), Y_test.reshape(-1)


    # initialize the geneticOptimizer   自定義種群數量和 最大迭代數
    cb_Optimizer = Genetic_geatpy.geneticOptimizer(NIND = 50, MAXGEN = 1000)
    #
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test ,os.path.join("./regression/" ,"catboost"),ensemble_model = "catboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "xgboost") ,ensemble_model = "xgboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "lightgbm"),ensemble_model="lightgbm")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "Adaboost"),ensemble_model="Adaboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "SVR")      ,ensemble_model="SVR")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "gbdt"), ensemble_model="gbdt")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "bagging") , ensemble_model="bagging")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "ExtraTrees"), ensemble_model="ExtraTrees")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "RandomForest"),ensemble_model="RandomForest")
    #



