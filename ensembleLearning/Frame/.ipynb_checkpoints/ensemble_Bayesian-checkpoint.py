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
import BLS

from sklearn.model_selection import KFold # http://cqtech.online/article/2021/9/19/28.html
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


    # print("mse: {},rmse: {},mae: {},r2: {},mad: {},mape: {},r2_adjusted: {},rmsle: {}".format(mse,rmse,mae,r2,mad,mape,r2_adjusted,rmsle))
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
    # print('Save the value of prediction successfully!!')
    
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
    
    def run(self,init_points = 40,n_iter = 60):
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
            
        elif(self.ensemble_model == "lightgbm"):
            optimazationFun = self.Lightgbm_cv_bayesianOpt
            optimazationPara =    {'boosting_type':(0, 3+1),
                            'n_estimators':(50, 2000),
                            'learning_rate':(1e-6, 1),
                            'subsample':(0.5, 1),
                            'subsample_freq':(0,4),
                            'colsample_bytree':(0.5,1),
                            'reg_alpha':(0,2),
                            'reg_lambda':(0,2)}
            
        elif (self.ensemble_model == "Adaboost"):
            optimazationFun = self.AdaBoost_cv_bayesianOpt
            optimazationPara =     {'n_estimators':(50,2000),
                             'learning_rate':(1e-6,1),
                             'loss':(0,3),
                             'criterion':(0,2),
                             'splitter':(0,2)
                            }
           
        elif (self.ensemble_model == "SVR"):
            optimazationFun = self.SVR_cv_bayesianOpt
            optimazationPara =    {'kernel':(0,4),
                             'degree':(2,13),
                             'gamma':(0,2),
                            }
            
        elif (self.ensemble_model == "gbdt"):
            optimazationFun = self.gbdt_cv_bayesianOpt
            optimazationPara = {'n_estimators':(50,2000),
                            'learning_rate':(1e-6,1),
                            'subsample':(0.5,1)
                            }
            
        elif (self.ensemble_model == "RandomForest"):
            optimazationFun = self.RandomForest_cv_bayesianOpt
            optimazationPara = {'n_estimators':(50,1000),
                             'min_samples_split':(2,4),
                             'min_samples_leaf':(1,4),
                             'oob_score':(0,2) }
            
        elif (self.ensemble_model == "ExtraTrees"):
            optimazationFun = self.ExtraTrees_cv_bayesianOpt
            optimazationPara =    {'n_estimators':(10,200),
                            'min_samples_split':(2,10)}
            
        
        elif (self.ensemble_model == "bagging"):
            optimazationFun = self.Bagging_cv_bayesianOpt
            optimazationPara =    {'n_estimators':(10,400),
                                    'max_samples':(0.5,1) ,
                                    'max_features':(0.5,1) ,   }

        elif (self.ensemble_model == "BLS"):
            optimazationFun = self.BLS_cv_bayesianOpt
            optimazationPara =    {'NumFea':(2,50),
                                    'NumWin':(2,50)
                                    'NumEnhan':(5,60)
                                    'S':(0.4,6)
                                    'C':(0,5)},
        
        elif (self.ensemble_model == "LogisticR"):
            optimazationFun = self.LogisticR_cv_bayesianOpt
            optimazationPara = {"penalty":(0,2),"tol":(1e-5,1e-2),"C":(0.5,2.5)}

        elif (self.ensemble_model == "GPR"):
            optimazationFun = self.GPR_cv_bayesianOpt
            optimazationPara = {"alpha":(1e-10,1e-5),"normalize_y":(0,2)}

            
        elif (self.ensemble_model == "BayesianRidge"):
            optimazationFun = self.BayesianRidge_cv_bayesianOpt
            optimazationPara = {"n_iter":(100,1000),"tol":(1e-4,1e-2),"alpha_1":(1e-6,1e-2),"alpha_2":(1e-6,1e-2),"lambda_1":(1e-6,1e-2),"lambda_2":(1e-6,1e-2),"normalize":(0,2)}
            
            
        elif (self.ensemble_model == "PAR"):
            optimazationFun = self.PAR_cv_bayesianOpt
            optimazationPara = {"C":(0.5,2.5),"tol":(1e-4,1e-2)}
 
            
        elif (self.ensemble_model == "Lr_Sgd"):
            optimazationFun = self.Lr_Sgd_cv_bayesianOpt
            optimazationPara = {"loss":(0,4),"penalty":(0,3),"alpha":(1e-6,1e-2),"l1_ratio":(0.01,0.6),"tol":(1e-4,1e-2),"learning_rate":(0,4),"eta0":(1e-4,1e-2),"power_t":(0.1,0.5)}
            
        elif (self.ensemble_model == "DecisionTree"):
            optimazationFun = self.DecisionTree_cv_bayesianOpt
            optimazationPara = {"splitter":(0,2),"min_samples_split":(2,6),"min_samples_leaf":(1,5)}
            
        elif (self.ensemble_model == "LinearSvr"):
            optimazationFun = self.LinearSvr_cv_bayesianOpt
            optimazationPara = {"tol":(1e-6,1e-2),"C":(0.01,1.0),"loss":(0,2)}
       
        elif (self.ensemble_model == "KNN"):
            optimazationFun = self.KNN_cv_bayesianOpt
            optimazationPara = {"n_neighbors":(3,10),"weights":(0,2),"leaf_size":(15,45),"P":(0,1)}
                       
        else:
            print("model name have error")
            return 0
        
        object_bo = BayesianOptimization(optimazationFun,optimazationPara)
    
        # 开始优化
        object_bo.maximize(init_points = init_points,n_iter = n_iter)
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
        
        if kFold <= 1:
#             print("不适用 交叉验证")
            kFold = 1 
            Regressor.fit(X_train,y_train)

        else: 
#             print("使用 交叉验证")
            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
            
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
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i

 
    
    # ===============  Lightgbm =================
    # boosting_type = ["gbdt","dart","goss","rf"]  # rf 還是會報錯 所以先沒放上去
    # n_estimators = [50,100,200,300,400,500,600,700,800,900,1000,1200,1500]
    # learning_rate = [0.01,0.05,0.1,0.15,0.2]
    # subsample = [1.0,0.8,0.6]
    # subsample_freq = [0,1,2,3]
    # colsample_bytree = [1,0.8,0.6]
    # reg_alpha = [0,1,2]
    # reg_lambda = [0,1,2]

    def Lightgbm_cv_bayesianOpt(self, boosting_type, n_estimators, learning_rate, subsample, subsample_freq, colsample_bytree, reg_alpha, reg_lambda):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        boosting_type_list = ["gbdt", "dart", "goss", "rf"]

        boosting_type = boosting_type_list[int(boosting_type)]
        n_estimators = int(n_estimators)
        subsample_freq = int(subsample_freq)
        
   
    
        # 模型参数设置
        parameterDict = {"boosting_type": boosting_type, "n_estimators": n_estimators,
                     "learning_rate": learning_rate,"subsample": subsample,
                      "colsample_bytree": colsample_bytree,"max_depth":-1,
                     "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,"objective":'regression',"random_state":random_seed}

        # 实例化模型 并训练模型
        Regressor = lgb.LGBMRegressor(**parameterDict)

        
        
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    

    # ===============  AdaBoost ==========#
    # n_estimators = [50,100,200,300,400,500,600,700,800]
    # learning_rate = [0.1,0.5,1,1.5,2]
    # loss = ["linear","square","exponential"]
    # criterion = ["mse","mae"]
    # splitter = ["best","random"]
    # # max_features = ["None"]
    # # max_leaf_nodes = ["None"]
    # # min_samples_split = [2]
    # # min_samples_leaf = [1]
    
    def AdaBoost_cv_bayesianOpt(self,n_estimators,learning_rate,loss,criterion,splitter):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        lossList = ["linear", "square", "exponential"]
        criterionList = ["mse", "mae"]
        splitterList = ["best", "random"]

        n_estimators = int(n_estimators)
        loss = lossList[int(loss)]
        criterion = criterionList[int(criterion)]
        splitter = splitterList[int(splitter)]

        # 模型参数设置
        parameterDict = {"n_estimators": n_estimators, "learning_rate": learning_rate,"loss": loss,
                     "criterion": criterion,"splitter": splitter,"max_features":"None",
                     "max_leaf_nodes": "None", "min_samples_split": 2,"min_samples_leaf":1}
        # 实例化模型 并训练模型
        Regressor = AdaBoostRegressor(n_estimators=n_estimators, learning_rate = learning_rate, loss = loss,
                            base_estimator=DecisionTreeRegressor(min_samples_split=2,min_samples_leaf=1,
                            splitter=splitter, criterion=criterion ))
        
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    # ===============  SVR =================
    # kernel = ["rbf","linear","poly","sigmoid"]
    # degree = [2,3,4,5,6,7,8,9,10,11,12]
    # gamma = ["auto","scale"]

    
    def SVR_cv_bayesianOpt(self,kernel,degree,gamma):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        kernelList = ["rbf","linear","poly","sigmoid"]
        gammaList  = ["auto","scale"]

    
        kernel = kernelList[int(kernel)]
        degree = int(degree)
        gamma = gammaList[int(gamma)]

        # if (kernel == "poly"):
        #     gamma = "scale"

        # 模型参数设置
        parameterDict = {"kernel": kernel, "degree": degree,"gamma": gamma}
        # 实例化模型 并训练模型
        Regressor = SVR(**parameterDict)

        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i


    # ===============  gbdt =================
    # random_state = 17
    # n_estimators = [50,100,150,200,250,300]
    # learning_rate = [0.05,0.1,0.15,0.2]
    # loss = ["ls"]
    # subsample = [1,0.8,0.6]
    # min_samples_split = [2]
    # max_depth = [3]
    # min_samples_leaf = [1]

    
    def gbdt_cv_bayesianOpt(self,n_estimators,learning_rate,subsample):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        n_estimators = int(n_estimators)

        # 模型参数设置
        parameterDict = {"n_estimators": n_estimators, "learning_rate": learning_rate,"subsample": subsample}
        # 实例化模型 并训练模型
        Regressor = GradientBoostingRegressor(**parameterDict)
        
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    

    # ===============  ExtraTrees =================
    # n_estimators = [10, 30, 50, 70, 90, 110, 130]
    # min_samples_split = [2, 4, 6, 8]
 

    
    def ExtraTrees_cv_bayesianOpt(self,n_estimators,min_samples_split):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        n_estimators = int(n_estimators)
        min_samples_split = int(min_samples_split)

        # 模型参数设置
        parameterDict = {"n_estimators": n_estimators, "min_samples_split": min_samples_split}
        # 实例化模型 并训练模型
        Regressor = ExtraTreesRegressor(**parameterDict)
        
        
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    

    # ===============  Lightgbm =================
    # boosting_type = ["gbdt","dart","goss","rf"]  # rf 還是會報錯 所以先沒放上去
    # n_estimators = [50,100,200,300,400,500,600,700,800,900,1000,1200,1500]
    # learning_rate = [0.01,0.05,0.1,0.15,0.2]
    # subsample = [1.0,0.8,0.6]
    # subsample_freq = [0,1,2,3]
    # colsample_bytree = [1,0.8,0.6]
    # reg_alpha = [0,1,2]
    # reg_lambda = [0,1,2]

    def Lightgbm_cv_bayesianOpt(self,boosting_type,n_estimators,learning_rate,subsample,subsample_freq,reg_alpha,reg_lambda):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        boosting_type_list = ["gbdt", "dart", "goss", "rf"]

        boosting_type = boosting_type_list[int(boosting_type)]
        n_estimators = int(n_estimators)
        subsample_freq = int(subsample_freq)
        
   
    
        # 模型参数设置
        parameterDict = {"boosting_type": boosting_type, "n_estimators": n_estimators,
                     "learning_rate": learning_rate,"subsample": subsample,
                      "colsample_bytree": colsample_bytree,"max_depth":-1,
                     "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,"objective":'regression',"random_state":random_seed}

        # 实例化模型 并训练模型
        Regressor = lgb.LGBMRegressor(**parameterDict)

        
        
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i

    
    
     


    # ===============  RandomForest =================
    # n_estimators = [50,100,200,300,400,500,600,700,800]
    # criterion = ["mse"]
    # max_features = ["None"]
    # max_leaf_nodes = ["None"]
    # min_samples_split = [2,3]
    # min_samples_leaf = [1,2,3]
    # oob_score = ["True","False"]
    # random_state = 17
    # n_jobs = -1

    def RandomForest_cv_bayesianOpt(self,n_estimators,min_samples_split,min_samples_leaf,oob_score):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        oob_score_List = ["True", "False"]
        n_estimators = int(n_estimators)
        min_samples_split = int(min_samples_split)
        min_samples_leaf = int(min_samples_leaf)
        oob_score = oob_score_List[int(oob_score)]

        # 模型参数设置
        parameterDict = {"n_estimators": n_estimators, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "oob_score": oob_score}
        # 实例化模型 并训练模型
        Regressor = RandomForestRegressor(**parameterDict)
        
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    

    # ===============  Bagging =================
#   n_estimators = [10+(5*i) for i in range(200)]
#   max_samples = [0.7,0.8,0.9,1.0]
#   max_features = [0.7,0.8,0.9,1.0]
#   warm_start 热启动 这个参数用于从上一次训练的结果的基础上再次进行训练
    def Bagging_cv_bayesianOpt(self,n_estimators,max_samples,max_features):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        
        # 模型参数设置
        parameter = {"warm_start":False,"n_estimators":int(n_estimators), "random_state":random_seed,"max_samples":max_samples,"max_features":max_features}
        # 实例化模型 并训练模型
        model = BaggingRegressor(**parameter)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 
            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
   


    # ===============  board learning system =================
    #     NumFea = [i for i in range(2,40,4)]
    #     NumWin = [i for i in range(5,40,5)]
    #     NumEnhan = [i for i in range(5,60,10)]
    #     S = [0.4,0.6,0.8,1,1.2,4]
    #     C = [2**-30,2**-10,2**-20,2**-40,1**-30] （0,5）
    
    def BLS_cv_bayesianOpt(self,S,C,NumFea,NumWin,NumEnhan):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        C = [2**-30,2**-10,2**-20,2**-40,1**-30][int(C)]
        # 模型参数设置
        parameter = {"s":S, "C":C, "NumFea":NumFea, "NumWin":NumWin, "NumEnhan":NumEnhan}
        # 实例化模型 并训练模型
        Regressor = BLS.BLSregressor(s=s, C=c, NumFea=nf, NumWin=nw, NumEnhan=ne)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
     

        
        
        
      

    # ===============  KNN =================
    #     n_neighbors = [3,5,7,9] # 默认为5
    #     weights = ['uniform', 'distance']
    #     leaf_size = [25,30,35] #默认是30
    #     P = [1,2] # 只在 wminkowski 和 minkowski 调 0,1
    
    def KNN_cv_bayesianOpt(self,n_neighbors,weights,leaf_size,P):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
       
        weights = ['uniform', 'distance'][int(weights)]
        P = [1,2][int(P)]
        # 模型参数设置
        paprameter = {"n_neighbors"=int(n_neighbors),"leaf_size"=int(leaf_size),"p"=int(P),"weights"=weights}
        # 实例化模型 并训练模型
        Regressor = KNeighborsRegressor(**paprameter)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
        
    # ===============  LinearSvr =================
    #      tol = [1e-5,1e-4,1e-3]
    #      C = [1.0,1.5,0.5,2.0,0.01]
    #      loss = ["epsilon_insensitive","squared_epsilon_insensitive"] (0,2)

    
    def LinearSvr_cv_bayesianOpt(self):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        loss = ["epsilon_insensitive","squared_epsilon_insensitive"][int(loss)]
        # 模型参数设置
        parameter = {"tol":tol,"C":C,"loss":loss,"random_state":random_seed}
        # 实例化模型 并训练模型
        Regressor = LinearSVR()
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    
    # ===============  DecisionTree =================
    # splitter = ["best","random"]
    # min_samples_split = [2,3,4,5]
    # min_samples_leaf = [1,2,3]
    # random_state = 17
    
    def DecisionTree_cv_bayesianOpt(self,splitter,min_samples_split,min_samples_leaf):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        splitter = ["best","random"][int(splitter)]
        # 模型参数设置
        parameter = {"criterion":"mse",
                     "splitter":splitter,
                     "min_samples_leaf":int(min_samples_leaf),
                     "min_samples_split":int(min_samples_split),
                     "random_state":random_seed}
        # 实例化模型 并训练模型
        Regressor = DecisionTreeRegressor(**parameter)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    
    
    
    # ===============  Lr_Sgd =================
    #     loss = ["squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"]
    #     penalty = ["l2","l1","elasticnet"]  
    #     alpha = [0.0001,0.001,0.0005,0.01]
    #     l1_ratio = [0.15,0.1,0.2,0.3,0.01], default=0.5
    #                 Elastic-Net（弹性网）混合参数，取值范围0 <= l1_ratio <= 1。
    #                 仅在penalty='elasticnet'时使用。
    #                 设置l1_ratio=0等同于使用L2惩罚，而设置l1_ratio=1等同于使用L1惩罚。
    #                 对于0 < l1_ratio <1，惩罚是L1和L2的组合。
    #     tol = [1e-3,1e-2,1e-4]
    #     learning_rate = ["constant","optimal","invscaling","adaptive"]
    #     eta0 = [0.01,0.015,0.005]
    #     power_t = [0.25,0.2,0.3]
    #     random_state = 17
    
    def Lr_Sgd_cv_bayesianOpt(self,l1_ratio,loss,penalty,alpha,tol,learning_rate,eta0,power_t):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        loss = ["squared_loss","huber","epsilon_insensitive","squared_epsilon_insensitive"][int(loss)]
        penalty = ["l2","l1","elasticnet"][int(penalty)]
        learning_rate =  ["constant","optimal","invscaling","adaptive"][int(learning_rate)]
        
        # 当penalty != 'elasticnet'时      设定 L1_ratio 为默认值
        if penalty  != 'elasticnet':
            L1_ratio = 0.5
        # 模型参数设置
        parameter = {"random_state"=random_seed,
                     "warm_start"=False,
                     "l1_ratio"=l1,
                     "loss"=l,
                     "penalty"=p,
                     "alpha"=a,
                     "tol"=t,
                     "learning_rate"=lr,
                     "eta0"=e,
                     "power_t"=pt}
        # 实例化模型 并训练模型
        model = SGDRegressor(**parameter)
                                                
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    
    # ===============  Passive Aggressive Regressor =================
    #     C = [1.0,0.5,1.5,2.0]
    #     tol = [1e-3,1e-2,1e-4]
    def PAR_cv_bayesianOpt(self,C,tol):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        # 模型参数设置
        parameter = {"C":C,"tol":tol,"random_state" = random_seed}
        # 实例化模型 并训练模型
        Regressor = PassiveAggressiveRegressor(**parameter)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    
    
    
    
    
    
    # ===============  BayesianRidge =================
    #     n_iter = [100,200,300,400,500]
    #     tol = [1e-3,2e-3,1e-4,1e-2]
    #     alpha_1 = [1e-6,1e-4,1e-5]
    #     alpha_2 = [1e-6,1e-4,1e-5]
    #     lambda_1 = [1e-6,1e-4,1e-5]
    #     lambda_2 = [1e-6,1e-4,1e-5]
    #     normalize = [True,False]
    
    def BayesianRidge_cv_bayesianOpt(self,n_iter,tol,alpha_1,alpha_2,lambda_1,lambda_2,normalize):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        normalize = [True,False][int(normalize)]
       
        # 模型参数设置
        parameter = {"n_iter"=int(n_iter),
                     "tol"=tol,
                     "alpha_1"=alpha_1,
                     "alpha_2"=alpha_2,
                     "lambda_1"=lambda_1,
                     "lambda_2"=lambda_2,
                     "normalize"=normalize}
        # 实例化模型 并训练模型
        Regressor = BayesianRidge(**parameter)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    
     
    
    # ===============  Gaussian Process Regressor =================
    #     alpha = [1e-10,1e-9,1e-11,1e-8]
    #     normalize_y = [True,False]

    def GPR_cv_bayesianOpt(self,alpha,normalize_y):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        normalize_y = [True,False][int(normalize_y)]
        
        # 模型参数设置
        parameter  = {alpha=alpha,normalize_y=normalize_y,random_state=random_seed}
        # 实例化模型 并训练模型
        Regressor = GaussianProcessRegressor(**parameter)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    
    # ===============  LogisticRegression =================
    #     penalty = ["l1","l2"]
    #     tol = [1e-3,1e-4,1e-5]
    #     C = [0.5,1,1.5,2.0]
    
    def LogisticR_cv_bayesianOpt(self,penalty,tol,C):
        X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
        
        penalty = ["l1","l2"][int(penalty)]
       
        # 模型参数设置
        parameter = {"penalty":penalty,"tol":tol,"C":C , "warm_start":False , "random_state":random_seed , "n_jobs":-1 , "solver" :'liblinear'}
        # 实例化模型 并训练模型
        Regressor = LogisticRegression(**parameter)
        # 如果kFold数值大于1 则启用 k-fold cross-validation
        if kFold <= 1:
            kFold = 1 
            Regressor.fit(X_train,y_train)
        else: 

            folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
                Regressor.fit(X_train[trn_idx],y_train[trn_idx])
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
        joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
        return ObjV_i
    
    
    

    
################################
# -模型定义模板如下              
################################
    
#     # ===============  xxxxx =================

      
#     def xxxxx_cv_bayesianOpt(self):
#         X_train, X_test, y_train, y_test,random_seed,save_path,kFold = self.getDatasetValues()
          
       
#         # 模型参数设置
       
#         # 实例化模型 并训练模型
       
#         # 如果kFold数值大于1 则启用 k-fold cross-validation
#         if kFold <= 1:
#             kFold = 1 
#             Regressor.fit(X_train,y_train)
#         else: 

#             folds = KFold(n_splits=kFold, shuffle=True, random_state=random_seed)
#             for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
#                 Regressor.fit(X_train[trn_idx],y_train[trn_idx])
#         # 计算模型在测试集上的效果
#         y_pre = Regressor.predict(X_test)
#         regDict = reg_calculate(y_test, y_pre,X_test.shape[0], X_test.shape[1])
#         ObjV_i = regDict["r2"]

#         # 模型保存
#         resultTitle = []
#         resultList = []

#         for Title, Parameter in parameterDict.items():
#             resultTitle.append(Title)
#             resultList.append(str(Parameter))

#         for Title, Parameter in regDict.items():
#             resultTitle.append(Title)
#             resultList.append(str(Parameter))

    
#         count = save_results(resultTitle, resultList, y_test, y_pre, save_path)
#         # 保存模型
#         model_path = os.path.join(save_path, 'Model')
#         if not os.path.exists(model_path):
#             os.makedirs(model_path)
#         joblib.dump(Regressor, os.path.join(model_path, str(count) + ".pkl"))
#         return ObjV_i
    