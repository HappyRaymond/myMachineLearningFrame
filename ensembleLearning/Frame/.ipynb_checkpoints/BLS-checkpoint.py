import numpy  as np
from numpy import random
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,SGDRegressor,BayesianRidge,LogisticRegression,PassiveAggressiveRegressor,ElasticNet,Ridge,Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
import joblib
from sklearn.preprocessing import MaxAbsScaler,StandardScaler,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
class BLSregressor:
    def __init__(self, s, C, NumFea, NumWin, NumEnhan):
        self.s = s
        self.C = C
        self.NumFea = NumFea
        self.NumEnhan = NumEnhan
        self.NumWin = NumWin

    def shrinkage(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    def tansig(self, x):
        return (2 / (1 + np.exp(-2 * x))) - 1

    def pinv(self, A, reg):
        return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

    def sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = np.dot(A.T, A)
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m, n], dtype='double')
        ok = np.zeros([m, n], dtype='double')
        uk = np.zeros([m, n], dtype='double')
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.dot(np.dot(L1, A.T), b)
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + np.dot(L1, tempc)
            ok = self.shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def fit(self, train_x, train_y):
        train_y = train_y.reshape(-1, 1)
        u = 0
        WF = list()
        for i in range(self.NumWin):
            random.seed(i + u)
            WeightFea = 2 * random.randn(train_x.shape[1] + 1, self.NumFea) - 1
            WF.append(WeightFea)
        random.seed(100)
        WeightEnhan = 2 * random.randn(self.NumWin * self.NumFea + 1, self.NumEnhan) - 1
        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
        y = np.zeros([train_x.shape[0], self.NumWin * self.NumFea])
        WFSparse = list()
        distOfMaxAndMin = np.zeros(self.NumWin)
        meanOfEachWindow = np.zeros(self.NumWin)
        for i in range(self.NumWin):
            WeightFea = WF[i]
            A1 = H1.dot(WeightFea)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            WeightFeaSparse = self.sparse_bls(A1, H1).T
            WFSparse.append(WeightFeaSparse)

            T1 = H1.dot(WeightFeaSparse)
            meanOfEachWindow[i] = T1.mean()
            distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
            y[:, self.NumFea * i:self.NumFea * (i + 1)] = T1
        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
        T2 = H2.dot(WeightEnhan)
        T2 = self.tansig(T2)
        T3 = np.hstack([y, T2])
        WeightTop = self.pinv(T3, self.C).dot(train_y)
        self.WeightTop = WeightTop
        self.WFSparse = WFSparse
        self.meanOfEachWindow = meanOfEachWindow
        self.distOfMaxAndMin = distOfMaxAndMin
        self.WeightEnhan = WeightEnhan
        return self

    def predict(self, test_x):
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
        yy1 = np.zeros([test_x.shape[0], self.NumWin * self.NumFea])
        for i in range(self.NumWin):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1[:, self.NumFea * i:self.NumFea * (i + 1)] = TT1
        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
        TT2 = self.tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1, TT2])
        NetoutTest = TT3.dot(self.WeightTop)
        NetoutTest = np.array(NetoutTest).reshape(1, -1)
        return NetoutTest

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=False):
        return {
            's': self.s,
            'C': self.C,
            'NumFea': self.NumFea,
            'NumWin': self.NumWin,
            'NumEnhan': self.NumEnhan
        }