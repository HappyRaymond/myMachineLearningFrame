# library
# standard library
import os
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

import time
from sklearn import metrics

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score,mean_squared_log_error

# 二維數據的回歸器 ANN CNN_1D

class General_Regression_Training():

    # 給優化函數判斷模型效果用的
    def fitness(evaluationStr="r2"):
        if (evaluationStr == "r2"):
            return self.r2
        elif (evaluationStr == "r2_adjusted"):
            return self.r2_adjusted
        elif (evaluationStr == "rmsle"):
            return self.rmsle
        elif (evaluationStr == "mape"):
            return self.mape
        elif (evaluationStr == "r2_adjusted"):
            return self.r2_adjusted
        elif (evaluationStr == "mad"):
            return self.mad
        elif (evaluationStr == "mae"):
            return self.mae
    # 保存参数  预测值 真实值 图片
    def save_results(self):
        # , resultTitle, resultList, y_test, test_prediction, save_path
        resultTitle     = [str(line) for line in self.resultDict.keys()]
        resultList      = [ "_".join([ str(l) for l in line]) if isinstance(line,list) else str(line) for line in self.resultDict.values()]
        y_test          = self.y_test
        test_prediction = self.test_prediction
        save_path       = self.save_path

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
        # 保存 train loss 和 test loss
        Loss_path = os.path.join(save_path, 'Loss')
        if not os.path.exists(Loss_path):
            os.makedirs(Loss_path)

        save_Loss = os.path.join(Loss_path, str(count) + '.csv')

        df = pd.DataFrame()
        df["TrainLoss"] = self.TrainLosses
        df["TestLoss"] = self.TestLosses
        df.to_csv(save_Loss, index=False)
        # 保存 prediction
        pred_path = os.path.join(save_path, 'Prediction')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        save_prediction = os.path.join(pred_path, str(count) + '.csv')
        df = pd.DataFrame()

        df["y_test"] = [i for i in y_test]
        df["test_prediction"] =[i for i in test_prediction]
        df.to_csv(save_prediction, index=False)

        print('Save the value of prediction successfully!!')

        # save the model weight
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if(self.use_more_gpu):
            torch.save(self.net.state_dict(), os.path.join(model_path, str(count) + ".pth"))
        else:
            torch.save(self.net.state_dict(), os.path.join(model_path, str(count) + ".pth"))

        return count


    def reg_calculate(self,y_true, y_predict,test_sample_size,feature_size):
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
        return mse, rmse, mae, r2, mad, mape, r2_adjusted, rmsle

    def __init__(self,net,learning_rate = [1e-3,1e-5,1e-7], batch_size = 1024, epoch = 2000, use_more_gpu = False,weight_decay=1e-8, device=0 ,save_path='CNN_Result'):

        self.net = net
        self.resultDict = {"learning_rate":learning_rate,"batch_size":batch_size,"epoch":epoch,"weight_decay":weight_decay,"use_more_gpu":use_more_gpu,"device":device,}
        self.resultDict = dict(self.resultDict,**self.net.feature())

        self.batch_size = batch_size
        self.use_more_gpu = use_more_gpu
        self.lr = learning_rate
        self.epoch = epoch
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch
        
        self.save_path = save_path  # 设置一条保存路径，直接把所有的值都收藏起来
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.avgLossList = []  # put the avgLoss data
        self.TrainLosses = []
        self.TestLosses = []
        self.t = 0
        self.D = []
        self.n = 0  # 来记录 梯度衰减 的次数
        self.limit = [1e-5, 1e-6, 1e-7]
        
    # 創建數據生成器
    def create_batch_size(self, X_train, y_train):

        datasets = torch.FloatTensor(np.c_[X_train, y_train])
        batch_train_set = DataLoader(datasets, batch_size=self.batch_size, shuffle=True)
        return batch_train_set
    
    

    # 訓練函數
    def fit(self, X_train, y_train, X_test, y_test):
        ''' training the network '''
        # input the dataset and transform into dataLoad
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        

        batch_train_set = self.create_batch_size(X_train, y_train)
        
        save_result = os.path.join(self.save_path, 'Results.csv')
        try:
            count = len(open(save_result, 'rU').readlines())
        except:
            count = 1

        net_weight = os.path.join(self.save_path, 'Weight')
        if not os.path.exists(net_weight):
            os.makedirs(net_weight)

        net_path = os.path.join(net_weight, str(count) + '.pkl')
        net_para_path = os.path.join(net_weight, str(count) + '_parameters.pkl')
        
        
    
        # set the net use cpu or gpu
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Let's use GPU: {}".format(self.device))
        else:
            print("Let's use CPU")
            
            
        
        if self.use_more_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
            self.net = nn.DataParallel(self.net)
        self.net.to(device)

        # network change to train model 
        self.net.train()
        # set optimizer and loss function
        try:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr[0], weight_decay=self.weight_decay)
        except:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()
        print("")
        # Officially start training

        start = time.time() # 计算时间
        limit = self.limit[0]
        for e in range(self.epoch):

            tempLoss = []
            # 訓練模式
            self.net.train()
            for b_train in batch_train_set:
                if torch.cuda.is_available():
                    # print('cuda')
                    # self.net = self.net.cuda()
                    train_x = Variable(b_train[:, :-1]).to(device)
                    train_y = Variable(b_train[:, -1:]).to(device)
                else:
                    train_x = Variable(b_train[:, :-1])
                    train_y = Variable(b_train[:, -1:])


                prediction = self.net(train_x)
                
                loss = criterion(prediction, train_y)
                tempLoss.append(float(loss))
                
                optim.zero_grad()
                loss.backward()
                optim.step()

            self.D.append(loss.cpu().data.numpy())
            avgloss =  np.array(tempLoss).sum() / len(tempLoss)
            self.avgLossList.append(avgloss)
            
            
            if( ( e + 1 ) % 10 == 0):
                print('Training... epoch: {}, loss: {}'.format((e + 1), self.avgLossList[-1]))

                self.net.eval()
                if torch.cuda.is_available():
                    test_x = Variable(torch.FloatTensor(self.X_test)).to(device)
                    test_y = Variable(torch.FloatTensor(self.y_test)).to(device)
                else:
                    test_x = Variable(torch.FloatTensor(self.X_test))
                    test_y = Variable(torch.FloatTensor(self.y_test))

                test_prediction = self.net(test_x)
                test_loss = criterion(test_prediction, test_y)

                self.TrainLosses.append(avgloss)
                self.TestLosses.append(test_loss.cpu().data.numpy())

                self.test_prediction = test_prediction.cpu().data.numpy()
                self.test_prediction[self.test_prediction < 0] = 0
                # self.mse, self.rmse, self.mae, self.mape, \
                #     self.r2, self.r2_adjusted, self.rmsle = self.reg_calculate(self.y_test, self.test_prediction  ,self.X_test.shape[-1] )


                #test_acc = self.__get_acc(test_prediction, test_y)
                # print('\033[1;35m Testing... epoch: {}, loss: {} , r2 {}\033[0m!'.format((e + 1), test_loss.cpu().data.numpy(), self.r2))

                
                
#                 plt.figure(figsize = (7,5))       #figsize是图片的大小`
#                 plt.plot( [i for  i in range(len(self.avgLossList))] ,self.avgLossList,'g-',label=u'Dense_Unet(block layer=5)')
#                 plt.legend()
#                 plt.xlabel(u'iters')
#                 plt.ylabel(u'loss')
#                 plt.title('Compare loss for different models in training')
#                 plt.show()
                
                
                
            
            # epoch 终止装置
            if len(self.D) >= 20:
                loss1 = np.mean(np.array(self.D[-20:-10]))
                loss2 = np.mean(np.array(self.D[-10:]))
                d = np.float(np.abs(loss2 - loss1)) # 計算loss的差值

                
                if d < limit or e == self.epoch-1  or e > (self.epoch-1)/3 * (self.n + 1)   : # 加入遍历完都没达成limit限定，就直接得到结果  
                    
                    self.D = []  # 重置
                    self.n += 1
                    print('The error changes within {}'.format(limit))
                    self.e = e + 1


                    #train_acc = self.__get_acc(prediction, train_y)
                    print(
                        'Training... epoch: {}, loss: {}'.format((e + 1), loss.cpu().data.numpy()))
                    
                    # torch.save(self.net.module.state_dict(), model_out_path) 多 GPU 保存
                    
                    torch.save(self.net, net_path)
                    torch.save(self.net.state_dict(), net_para_path)
                    
                    self.net.eval()
                    if torch.cuda.is_available():
                        test_x = Variable(torch.FloatTensor(self.X_test)).to(device)
                        test_y = Variable(torch.FloatTensor(self.y_test)).to(device)
                    else:
                        test_x = Variable(torch.FloatTensor(self.X_test))
                        test_y = Variable(torch.FloatTensor(self.y_test))
                    
                    test_prediction = self.net(test_x)
                    test_loss = criterion(test_prediction, test_y)

                
                    self.test_prediction = test_prediction.cpu().data.numpy()
                    self.test_prediction[self.test_prediction < 0] = 0
                    
                    print("self.y_test",self.y_test[0],np.array(self.y_test).shape)
                    print("self.test_prediction",self.test_prediction[0],self.test_prediction.shape)
                    print("self.X_test.shape[-1]",self.X_test.shape[-1])
                    
                    self.mse, self.rmse, self.mae, \
                    self.r2,self.mad, self.mape,  self.r2_adjusted, self.rmsle = self.reg_calculate(self.y_test, self.test_prediction  ,self.X_test.shape[0],self.X_test.shape[-1] )
                        
                    
                    #test_acc = self.__get_acc(test_prediction, test_y)
                    print('\033[1;35m Testing... epoch: {}, loss: {} , r2 {}\033[0m!'.format((e + 1), test_loss.cpu().data.numpy(), self.r2))
                    
                    # 已经梯度衰减了 2 次
                    if self.n == 3:
                        print('The meaning of the loop is not big, stop!!')
                        break
                    limit = self.limit[self.n]
                    print('Now learning rate is : {}'.format(self.lr[self.n]))
                    optim.param_groups[0]["lr"] = self.lr[self.n]
            
            
        end = time.time()
        self.t = end - start
        print('Training completed!!! Time consuming: {}'.format(str(self.t)))

        

        # 计算结果 mse, rmse, mae, r2, mad, mape, r2_adjusted, rmsle
        self.mse, self.rmse, self.mae, \
        self.r2,self.mad, self.mape,  self.r2_adjusted, self.rmsle = self.reg_calculate(self.y_test, self.test_prediction,self.X_test.shape[0],
                                                                    self.X_test.shape[-1])
        #
        resDict = {"mse":self.mse, "rmse":self.rmse, "mae":self.mae, "r2":self.r2, "r2_adjusted":self.r2_adjusted, "mad":self.mad, "mape":self.mape, "rmsle":self.rmsle}
        self.resultDict = dict(resDict,**self.resultDict)
    
        
        # 給優化函數判斷模型效果用的 
    def fitness(evaluationStr = "r2"):
        if(evaluationStr == "r2"):
            return self.r2
        elif(evaluationStr == "r2_adjusted"):
            return  self.r2_adjusted
        elif(evaluationStr == "rmsle"):
            return  self.rmsle
        elif(evaluationStr == "mape"):
            return  self.mape
        elif(evaluationStr == "r2_adjusted"):
            return  self.r2_adjusted
        elif(evaluationStr == "mad"):
            return  self.mad
        elif(evaluationStr == "mae"):
            return  self.mae


# 三維數據的回歸器 針對 transformer lstm
class General_Regression_Training_3d():

    # 給優化函數判斷模型效果用的
    def fitness(evaluationStr="r2"):
        if (evaluationStr == "r2"):
            return self.r2
        elif (evaluationStr == "r2_adjusted"):
            return self.r2_adjusted
        elif (evaluationStr == "rmsle"):
            return self.rmsle
        elif (evaluationStr == "mape"):
            return self.mape
        elif (evaluationStr == "r2_adjusted"):
            return self.r2_adjusted
        elif (evaluationStr == "mad"):
            return self.mad
        elif (evaluationStr == "mae"):
            return self.mae

    # 保存参数  预测值 真实值 图片
    def save_results(self):
        # , resultTitle, resultList, y_test, test_prediction, save_path
        resultTitle = [str(line) for line in self.resultDict.keys()]
        resultList = ["_".join([str(l) for l in line]) if isinstance(line, list) else str(line) for line in
                      self.resultDict.values()]
        y_test = self.y_test
        test_prediction = self.test_prediction
        save_path = self.save_path

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
        # 保存 train loss 和 test loss
        Loss_path = os.path.join(save_path, 'Loss')
        if not os.path.exists(Loss_path):
            os.makedirs(Loss_path)

        save_Loss = os.path.join(Loss_path, str(count) + '.csv')

        df = pd.DataFrame()
        df["TrainLoss"] = self.TrainLosses
        df["TestLoss"] = self.TestLosses
        df.to_csv(save_Loss, index=False)
        # 保存 prediction
        pred_path = os.path.join(save_path, 'Prediction')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        save_prediction = os.path.join(pred_path, str(count) + '.csv')
        df = pd.DataFrame()

        df["y_test"] = [i for i in y_test]
        df["test_prediction"] = [i for i in test_prediction]
        df.to_csv(save_prediction, index=False)

        print('Save the value of prediction successfully!!')

        # save the model weight
        model_path = os.path.join(save_path, 'Model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if (self.use_more_gpu):
            torch.save(self.net.state_dict(), os.path.join(model_path, str(count) + ".pth"))
        else:
            torch.save(self.net.state_dict(), os.path.join(model_path, str(count) + ".pth"))

        return count

        
        
    
     # 定义一个评估指标的函数评估模型好坏 
    def reg_calculate(self,y_true, y_predict,test_sample_size,feature_size):
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
        return mse, rmse, mae, r2, mad, mape, r2_adjusted, rmsle

    def __init__(self, net, learning_rate=[1e-3, 1e-5, 1e-7], batch_size=1024, epoch=2000, use_more_gpu=False,
                 weight_decay=1e-8, device=0, save_path='CNN_Result'):

        self.net = net
        self.resultDict = {"learning_rate": learning_rate, "batch_size": batch_size, "epoch": epoch,
                           "weight_decay": weight_decay, "use_more_gpu": use_more_gpu, "device": device, }
        self.resultDict = dict(self.resultDict, **self.net.feature())

        self.batch_size = batch_size
        self.use_more_gpu = use_more_gpu
        self.lr = learning_rate
        self.epoch = epoch
        self.weight_decay = weight_decay
        self.device = device
        self.epoch = epoch

        self.save_path = save_path  # 设置一条保存路径，直接把所有的值都收藏起来
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.avgLossList = []  # put the avgLoss data
        self.TrainLosses = []
        self.TestLosses = []
        self.t = 0
        self.D = []
        self.n = 0  # 来记录 梯度衰减 的次数
        self.limit = [1e-5, 1e-6, 1e-7]

    # 創建數據生成器
    def create_batch_size(self, X_train, y_train):
        p = np.random.permutation(X_train.shape[0])
        data = X_train[p]
        label = y_train[p]

        batch_size = self.batch_size
        batch_len = X_train.shape[0] // batch_size + 1

        b_datas = []
        b_labels = []
        for i in range(batch_len):
            try:
                batch_data = data[batch_size * i: batch_size * (i + 1)]
                batch_label = label[batch_size * i: batch_size * (i + 1)]
            except:
                batch_data = data[batch_size * i: -1]
                batch_label = label[batch_size * i: -1]
            b_datas.append(batch_data)
            b_labels.append(batch_label)

        return b_datas, b_labels

    # 訓練函數
    def fit(self, X_train, y_train, X_test, y_test):
        ''' training the network '''
        # input the dataset and transform into dataLoad
        # if y is a scalar
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        b_data, b_labels = self.create_batch_size(X_train, y_train)

        save_result = os.path.join(self.save_path, 'Results.csv')
        try:
            count = len(open(save_result, 'rU').readlines())
        except:
            count = 1

        net_weight = os.path.join(self.save_path, 'Weight')
        if not os.path.exists(net_weight):
            os.makedirs(net_weight)

        net_path = os.path.join(net_weight, str(count) + '.pkl')
        net_para_path = os.path.join(net_weight, str(count) + '_parameters.pkl')

        # set the net use cpu or gpu
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Let's use GPU: {}".format(self.device))
        else:
            print("Let's use CPU")

        if self.use_more_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            # dim = 0 [64, xxx] -> [32, ...], [32, ...] on 2GPUs
            self.net = nn.DataParallel(self.net)
        self.net.to(device)

        # network change to train model
        self.net.train()
        # set optimizer and loss function
        try:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr[0], weight_decay=self.weight_decay)
        except:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()
        print("")
        # Officially start training

        start = time.time()  # 计算时间
        limit = self.limit[0]
        for e in range(self.epoch):

            tempLoss = []
            # 訓練模式
            self.net.train()
            for i in range(len(b_data)):
                if torch.cuda.is_available():
                    # print('cuda')
                    # self.net = self.net.cuda()
                    train_x = Variable(torch.FloatTensor(b_data[i])).to(device)
                    train_y = Variable(torch.FloatTensor(b_labels[i])).to(device)
                else:
                    train_x = Variable(torch.FloatTensor(b_data[i]))
                    train_y = Variable(torch.FloatTensor(b_labels[i]))

                prediction = self.net(train_x)

                loss = criterion(prediction, train_y)
                tempLoss.append(float(loss))

                optim.zero_grad()
                loss.backward()
                optim.step()

            self.D.append(loss.cpu().data.numpy())
            avgloss = np.array(tempLoss).sum() / len(tempLoss)
            self.avgLossList.append(avgloss)

            if ((e + 1) % 100 == 0):
                print('Training... epoch: {}, loss: {}'.format((e + 1), self.avgLossList[-1]))

                self.net.eval()
                if torch.cuda.is_available():
                    test_x = Variable(torch.FloatTensor(self.X_test)).to(device)
                    test_y = Variable(torch.FloatTensor(self.y_test)).to(device)
                else:
                    test_x = Variable(torch.FloatTensor(self.X_test))
                    test_y = Variable(torch.FloatTensor(self.y_test))

                test_prediction = self.net(test_x)
                test_loss = criterion(test_prediction, test_y)

                self.TrainLosses.append(avgloss)
                self.TestLosses.append(test_loss.cpu().data.numpy())

                self.test_prediction = test_prediction.cpu().data.numpy()
                self.test_prediction[self.test_prediction < 0] = 0
                # self.mse, self.rmse, self.mae, self.mape, \
                #     self.r2, self.r2_adjusted, self.rmsle = self.reg_calculate(self.y_test, self.test_prediction  ,self.X_test.shape[-1] )

                # test_acc = self.__get_acc(test_prediction, test_y)
                # print('\033[1;35m Testing... epoch: {}, loss: {} , r2 {}\033[0m!'.format((e + 1), test_loss.cpu().data.numpy(), self.r2))

            #                 plt.figure(figsize = (7,5))       #figsize是图片的大小`
            #                 plt.plot( [i for  i in range(len(self.avgLossList))] ,self.avgLossList,'g-',label=u'Dense_Unet(block layer=5)')
            #                 plt.legend()
            #                 plt.xlabel(u'iters')
            #                 plt.ylabel(u'loss')
            #                 plt.title('Compare loss for different models in training')
            #                 plt.show()

            # epoch 终止装置
            if len(self.D) >= 20:
                loss1 = np.mean(np.array(self.D[-20:-10]))
                loss2 = np.mean(np.array(self.D[-10:]))
                d = np.float(np.abs(loss2 - loss1))  # 計算loss的差值

                if d < limit or e == self.epoch - 1 or e > (self.epoch - 1) / 3 * (
                        self.n + 1):  # 加入遍历完都没达成limit限定，就直接得到结果

                    self.D = []  # 重置
                    self.n += 1
                    print('The error changes within {}'.format(limit))
                    self.e = e + 1

                    # train_acc = self.__get_acc(prediction, train_y)
                    print(
                        'Training... epoch: {}, loss: {}'.format((e + 1), loss.cpu().data.numpy()))

                    # torch.save(self.net.module.state_dict(), model_out_path) 多 GPU 保存

                    torch.save(self.net, net_path)
                    torch.save(self.net.state_dict(), net_para_path)

                    self.net.eval()
                    if torch.cuda.is_available():
                        test_x = Variable(torch.FloatTensor(self.X_test)).to(device)
                        test_y = Variable(torch.FloatTensor(self.y_test)).to(device)
                    else:
                        test_x = Variable(torch.FloatTensor(self.X_test))
                        test_y = Variable(torch.FloatTensor(self.y_test))

                    test_prediction = self.net(test_x)
                    test_loss = criterion(test_prediction, test_y)

                    self.test_prediction = test_prediction.cpu().data.numpy()
                    self.test_prediction[self.test_prediction < 0] = 0

                    #                     print("self.y_test",np.array(self.y_test).shape)
                    #                     print("self.test_prediction",self.test_prediction.shape)
                    #                     print("self.test_prediction",self.test_prediction)
                    #                     print("self.X_test.shape[-1]",self.X_test.shape[-1])

                    self.mse, self.rmse, self.mae, \
                    self.r2,self.mad, self.mape,  self.r2_adjusted, self.rmsle = self.reg_calculate(self.y_test, self.test_prediction,self.X_test.shape[0],
                                                                               self.X_test.shape[-1])

                    # test_acc = self.__get_acc(test_prediction, test_y)
                    print('\033[1;35m Testing... epoch: {}, loss: {} , r2 {}\033[0m!'.format((e + 1),
                                                                                             test_loss.cpu().data.numpy(),
                                                                                             self.r2))

                    # 已经梯度衰减了 2 次
                    if self.n == 3:
                        print('The meaning of the loop is not big, stop!!')
                        break
                    limit = self.limit[self.n]
                    print('Now learning rate is : {}'.format(self.lr[self.n]))
                    optim.param_groups[0]["lr"] = self.lr[self.n]

        end = time.time()
        self.t = end - start
        print('Training completed!!! Time consuming: {}'.format(str(self.t)))

        

        # 计算结果
        self.mse, self.rmse, self.mae, \
        self.r2,self.mad, self.mape,  self.r2_adjusted, self.rmsle  = self.reg_calculate(self.y_test, self.test_prediction,self.X_test.shape[0],
                                                                   self.X_test.shape[-1])
        #   
        resDict = {"mse":self.mse, "rmse":self.rmse, "mae":self.mae, "r2":self.r2, "r2_adjusted":self.r2_adjusted, "mad":self.mad, "mape":self.mape, "rmsle":self.rmsle}
        self.resultDict = dict(resDict, **self.resultDict)
        # 給優化函數判斷模型效果用的

    def fitness(evaluationStr="r2"):
        if (evaluationStr == "r2"):
            return self.r2
        elif (evaluationStr == "r2_adjusted"):
            return self.r2_adjusted
        elif (evaluationStr == "rmsle"):
            return self.rmsle
        elif (evaluationStr == "mape"):
            return self.mape
        elif (evaluationStr == "r2_adjusted"):
            return self.r2_adjusted
        elif (evaluationStr == "mad"):
            return self.mad
        elif (evaluationStr == "mae"):
            return self.mae




