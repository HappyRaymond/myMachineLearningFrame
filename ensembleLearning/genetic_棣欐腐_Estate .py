from Frame import Genetic_geatpy # 导入库
from sklearn import datasets     # 波士頓房價數據集
import os
import pandas as pd
from sklearn import preprocessing
import numpy as np


# 按區域的
def myTrain_test_split_Estate(path):
    data = pd.read_csv(path)
    # estateList = list({}.fromkeys(data["Y3_year"].values.tolist()).keys())
    estate_Index = data.columns.tolist().index("Y1_Estate")
    trainData,testData = [],[]

    for line in data.values.tolist():
        if line[estate_Index] % 5 == 0 :
            print(line[estate_Index] )
            testData.append(line)
        else:
            trainData.append(line)
    npData = [ i for i in trainData]
    npData.extend(testData)
    npData = np.array(npData)
    X = npData[:, :len(npData[0]) - 1]
    y = npData[:, len(npData[0]) - 1:]

    zscore = preprocessing.StandardScaler()
    X_zscore = zscore.fit_transform(X)

    trainData = np.array(trainData)

    train_X = X_zscore[:trainData.shape[0]]
    test_X = X_zscore[trainData.shape[0]:]
    train_y = y[:trainData.shape[0]]
    test_y = y[trainData.shape[0]:]
    print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)
    return train_X, test_X, train_y, test_y


def myTrain_test_split(path, splitYear=2018):
    data = pd.read_csv(path)
    testData = data[data["Y3_year"] >= splitYear]
    trainData = data[data["Y3_year"] < splitYear]
    print(trainData.shape, testData.shape)


    Data = pd.concat([trainData, testData])
    print(Data["Y3_year"].head(200))

    npData = Data.values

    X = npData[:, :87]
    y = npData[:, 87:]
    zscore = preprocessing.StandardScaler()
    X_zscore = zscore.fit_transform(X)

    train_X = X_zscore[:trainData.shape[0]]
    test_X = X_zscore[trainData.shape[0]:]
    train_y = y[:trainData.shape[0]]
    test_y = y[trainData.shape[0]:]
    print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)
    return train_X, test_X, train_y, test_y




if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = myTrain_test_split("../dataset/BayesianRidge_economics_finished_香港.csv", splitYear=2019)
    Y_train,Y_test = Y_train.reshape(-1),Y_test.reshape(-1)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # initialize the geneticOptimizer   自定義種群數量和 最大迭代數
    cb_Optimizer = Genetic_geatpy.geneticOptimizer(NIND = 50, MAXGEN = 1000)
    #
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test ,os.path.join("./regression/" ,"catboost" + "2019"),ensemble_model = "catboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "xgboost"+ "2019") ,ensemble_model = "xgboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "lightgbm"+ "2019"),ensemble_model="lightgbm")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "Adaboost"+ "2019"),ensemble_model="Adaboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "SVR"+ "2019")      ,ensemble_model="SVR")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "gbdt"+ "2019"), ensemble_model="gbdt")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "bagging"+ "2019") , ensemble_model="bagging")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "ExtraTrees"+ "2019"), ensemble_model="ExtraTrees")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "RandomForest"+ "2019"),ensemble_model="RandomForest")
    #

    X_train, X_test, Y_train, Y_test = myTrain_test_split_Estate("../dataset/BayesianRidge_economics_finished_香港.csv")
    Y_train, Y_test = Y_train.reshape(-1), Y_test.reshape(-1)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # initialize the geneticOptimizer   自定義種群數量和 最大迭代數
    cb_Optimizer = Genetic_geatpy.geneticOptimizer(NIND=50, MAXGEN=1000)
    #
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "catboost" + "_Estate"),ensemble_model="catboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "xgboost" + "_Estate"),ensemble_model="xgboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "lightgbm" + "_Estate"),ensemble_model="lightgbm")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "Adaboost" + "_Estate"),ensemble_model="Adaboost")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "SVR" + "_Estate"), ensemble_model="SVR")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "gbdt" + "_Estate"), ensemble_model="gbdt")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "bagging" + "_Estate"),ensemble_model="bagging")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "ExtraTrees" + "_Estate"),ensemble_model="ExtraTrees")
    cb_Optimizer.run(X_train, X_test, Y_train, Y_test, os.path.join("./regression/", "RandomForest" + "_Estate"),ensemble_model="RandomForest")



