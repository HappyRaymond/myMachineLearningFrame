from sklearn import datasets  # 导入库
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import  Manager
from sklearn import metrics
import numpy as np

import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor

import pandas as pd
import numpy as np
from Frame import myDataset
from sklearn.preprocessing import StandardScaler
import os
import joblib
np.set_printoptions(suppress=True)

from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score,mean_squared_log_error

import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR

from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
################################
# 评估指标                      #
################################

def reg_calculate(y_true, y_predict,test_sample_size,feature_size):

    # try except 的原因是有时候有些结果不适合用某种评估指标
    try:
        mse = mean_squared_error(y_true, y_predict)
    except:
        mse = np.inf
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    except:
        rmse = np.inf
    try:
        mae = mean_absolute_error(y_true, y_predict)
    except:
        mae= np.inf
    try:
        r2 = r2_score(y_true, y_predict)
    except:
        r2 = np.inf
    try:
        mad = median_absolute_error(y_true, y_predict)
    except:
        mad = np.inf
    try:
        mape = np.mean(np.abs((y_true - y_predict) / y_true)) * 100
    except:
        mape = np.inf
    try:
        if (test_sample_size > feature_size):
            r2_adjusted = 1 - ((1 - r2) * (test_sample_size - 1)) / (test_sample_size - feature_size - 1)
    except:
        r2_adjusted = np.inf
    try:
        rmsle = np.sqrt(mean_squared_log_error(y_true, y_predict))
    except:
        rmsle = np.inf


    print("mse: {},rmse: {},mae: {},r2: {},mad: {},mape: {},r2_adjusted: {},rmsle: {}".format(mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle))
    return {"mse":mse, "rmse":rmse, "mae":mae, "r2":r2, "mad":mad, "mape":mape, "r2_adjusted":r2_adjusted, "rmsle":rmsle}



################################
# 保存实验结果                   #
# https://blog.csdn.net/qq_36523839/article/details/80707678
# https://blog.csdn.net/qq_32590631/article/details/82831613
################################


def save_results(resultTitle, resultList, y_test, test_prediction, save_path):
    # 预测值不能小于0  否则会报错
    test_prediction[test_prediction < 0] = 0
    
    # 计算行数，匹配 prediciton 的保存
    save_result = "/".join([save_path, 'result.csv'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        count = len(open(save_result, 'rU').readlines())
    except:
        count = 1

    # 判断是否存在未见 没有则写入文件 有则追加写入
    resultTitle.insert(0, "count")
    resultList.insert(0, str(count))

    if not os.path.exists(save_result):
        with open(save_result, 'w') as f:
            titleStr = ",".join(resultTitle)
            f.write(titleStr)
            f.write('\n')

    with open(save_result, 'a+') as f:
        contentStr = ",".join(resultList)
        f.write(contentStr)
        f.write('\n')

    # 保存 prediction
    pred_path = os.path.join(save_path, 'Prediction')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    save_prediction = os.path.join(pred_path, str(count) + '.csv')
    df = pd.DataFrame()
    df["y_test"] = y_test
    df["test_prediction"] = test_prediction
    df.to_csv(save_prediction, index=False)

    # np.savetxt(save_prediction, np.append(np.array(y_test), test_prediction, axis=1), delimiter=',')
    print('Save the value of prediction successfully!!')

    return count

################################
# 贝叶斯优化通用设置                 
################################



################################
# 模型优化函数
#先要定义一个目标函数。
# 比如此时，函数输入为随机森林的所有参数，输出为模型交叉验证5次的AUC均值，作为我们的目标函数。
# 因为bayes_opt库只支持最大值，所以最后的输出如果是越小越好，那么需要在前面加上负号，以转为最大值。
################################


# ===================   catboost =============#
# iterations = Vars[i, 0]  0-1000
# learning_rate = Vars[i, 1]
# depth = Vars[i, 2]
# l2_leaf_reg = int(Vars[i, 3])
# loss_function_index = int(Vars[i, 4])
# one_hot_max_size = int(Vars[i, 5])     # 0-2
class myBayesianoptimazation():
    def __init__(self,modelName,X_train, X_test, y_train, y_test,random_seed = 1998,save_path = "./ensembleLearning_hk/model/",kFold = 5 ):
        self.X_train, self.X_test, self.y_train, self.y_test,self.random_seed,self.save_path,self.kFold = \
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold 
        
        self.modelName = modelName
        
        
    def getDatasetValues(self):
        return self.X_train, self.X_test, self.y_train, self.y_test,self.random_seed,self.save_path,self.kFold
    
    def run(self):
        # 选择优化的函数和参数
        optimazationFun,optimazationPara = 0,0
        if (self.modelName == "catboost"):
            optimazationFun = self.catboost_cv_bayesianOpt
            optimazationPara = {'iterations':(200, 2000),
                              'learning_rate':(2e-6, 1),
                              'depth':(1, 15),
                              'loss_function_index':(1, 5),
                              'l2_leaf_reg':(0,2),
                              'one_hot_max_size':(0,2)}
        elif (self.modelName == "xgboost"):
            optimazationFun = self.XGBoost_cv_bayesianOpt
            optimazationPara = {'eta':(0,1),
                              'max_depth':(1,15),
                              'subsample':(0.2,1),
                              'reg_lambda':(0,5),
                              'reg_alpha':(0,5)
                            }
#         elif(self.ensemble_model == "lightgbm"):
            
#         elif (self.ensemble_model == "Adaboost"):
           
#         elif (self.ensemble_model == "SVR"):
            
#         elif (self.ensemble_model == "gbdt"):
            
#         elif (self.ensemble_model == "bagging"):
            
#         elif (self.ensemble_model == "RandomForest"):
           
#         elif (self.ensemble_model == "ExtraTrees"):
            
        else:
            print("model name have error")
            return 0
        print(optimazationFun)
        print(optimazationPara)
        
        object_bo = BayesianOptimization(optimazationFun,optimazationPara)
    
        # 开始优化
        object_bo.maximize(init_points = 40,n_iter = 60)
        # object_bo.max() # https://zhuanlan.zhihu.com/p/131222363

    def catboost_cv_bayesianOpt(self,iterations, learning_rate, depth, loss_function_index,l2_leaf_reg,one_hot_max_size):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        lossList = ["MAE","MAPE","Poisson","Quantile","RMSE","MultiRMSE"]
        # 模型参数设置
        parameterDict = {"iterations": int(iterations), "learning_rate": learning_rate, "depth": int(depth),
                         "loss_function": lossList[int(loss_function_index)],
                         "l2_leaf_reg": l2_leaf_reg, "one_hot_max_size": int(one_hot_max_size),"random_seed":int(random_seed),"task_type": "CPU", "logging_level": "Silent"}
        # 实例化模型 并训练模型
        Regressor = cb.CatBoostRegressor(**parameterDict)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        print("如果kFold数值大于1 则启用 k-fold cross-validation")
        if kFold <= 1:
            print("不适用 交叉验证")
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 
            print("使用 交叉验证")
            train_val = cross_val_score(Regressor,X_train, y_train, scoring='r2', cv=kFold).mean()
        
        # 计算模型在测试集上的效果
        y_pre = Regressor.predict(X_test)
        regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
        ObjV_i = regDict["r2"]

        # 模型保存
        resultTitle = []
        resultList = []

        for Title, Parameter in parameterDict.items():
            resultTitle.append(Title)
            resultList.append(str(Parameter))
            
        for Title, Parameter in regDict.items():
            resultTitle.append(Title)
            resultList.append(str(Parameter))


        count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        Regressor.save_model(os.path.join(model_path, str(count) + ".model"))

        return ObjV_i

    # ===============  XGBoost ==========#
    # eta  [0,1]
    # max_depth [1,∞]
    # subsample [default=1] 取值范围为：(0,1]
    # reg_lambda [default = 1] 权重的 L2 正则化项 取值范围为：(0,1]
    # reg_alpha [default = 0] 权重的 L1 正则化项 取值范围为：(0,1]
    
    def XGBoost_cv_bayesianOpt(self,eta,max_depth,subsample,reg_lambda,reg_alpha):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        # 模型参数设置
        parameterDict = {"eta": eta, "max_depth": max_depth, "subsample": subsample,"reg_lambda":reg_lambda,"reg_alpha":reg_alpha}
        # 实例化模型 并训练模型
        Regressor = xgb.XGBRegressor(**parameterDict)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_trains)
        else: 
            train_val = cross_val_score(Regressor,X_train, y_train, scoring='r2', cv=kFold).mean()
            print("train_val %{}"%train_val)
        # 计算模型在测试集上的效果
        y_pre = Regressor.predict(X_test)
        regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
        ObjV_i = regDict["r2"]

        # 模型保存
        resultTitle = []
        resultList = []

        for Title, Parameter in parameterDict.items():
            resultTitle.append(Title)
            resultList.append(str(Parameter))

        for Title, Parameter in regDict.items():
            resultTitle.append(Title)
            resultList.append(str(Parameter))

    
        count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
        # 保存模型
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        Regressor.save_model(os.path.join(model_path, str(count) + ".model"))

        return ObjV_i
