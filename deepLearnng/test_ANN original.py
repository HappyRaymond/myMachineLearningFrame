       from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn import metrics
import numpy as np
import catboost as cb

import numpy as np
import geatpy as ea
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import multiprocessing as mp
from multiprocessing import Manager
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
import geatpy as ea  # import geatpy


from frames.DeeplearningRegression import ANN




# ==========================   加載數據集 ==============================
def Shard_the_dataset(df, yearNum=2018):
    scaler = StandardScaler()
    scaler.fit(df.drop("Usable_Area_Unit_Price", axis=1))

    trainDataset = df.loc[df['year'] < yearNum]
    testDataset = df.loc[df['year'] >= yearNum]

    print(df.shape, trainDataset.shape, testDataset.shape)
    print(trainDataset.columns.tolist())
    X_train = trainDataset.drop("Usable_Area_Unit_Price", axis=1)
    y_train = trainDataset.Usable_Area_Unit_Price
    X_test = testDataset.drop("Usable_Area_Unit_Price", axis=1)
    y_test = testDataset.Usable_Area_Unit_Price

    npT = scaler.transform(X_train)
    dfT = pd.DataFrame(npT)
    pd.DataFrame(dfT)
    dfT.columns = X_train.columns
    X_train = dfT

    npT = scaler.transform(X_test)
    dfT = pd.DataFrame(npT)
    pd.DataFrame(dfT)
    dfT.columns = X_test.columns
    X_test = dfT

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


                      # 讀取 數據集 並標準化
def train_test_split():
    filePathList = myDataset.getFiles("../dataset", ".csv")

    for i in range(len(filePathList)):
        print(filePathList[i])
        if (filePathList.find("Economise") != -1  and filePathList.find("Hong Kong Island") != -1):
            Economise_HK_Data = pd.read_csv(filePathList[i])
        elif (filePathList.find("Economise") != -1  and filePathList.find("Kowloon") != -1):
            Economise_KL_Data = pd.read_csv(filePathList[i])
        elif (filePathList.find("Economise") != -1  and filePathList.find("New Territories") != -1):
            Economise_NT_Data = pd.read_csv(filePathList[i])
    dataDict = {"HK":Economise_HK_Data,"KL":Economise_KL_Data,"NT":Economise_NT_Data}

    for k,v in dataDict.items():
        # print(k)
        dataDict[k] = Shard_the_dataset(v, yearNum=2018)

    return dataDict






"""
该案例展示了如何利用进化算法+多进程/多线程来优化SVM中的两个参数：C和Gamma。
在执行本案例前，需要确保正确安装sklearn，以保证SVM部分的代码能够正常执行。
本函数需要用到一个外部数据集，存放在同目录下的iris.data中，
并且把iris.data按3:2划分为训练集数据iris_train.data和测试集数据iris_test.data。
有关该数据集的详细描述详见http://archive.ics.uci.edu/ml/datasets/Iris
在执行脚本main.py中设置PoolType字符串来控制采用的是多进程还是多线程。
注意：使用多进程时，程序必须以“if __name__ == '__main__':”作为入口，
      这个是multiprocessing的多进程模块的硬性要求。
"""
#  ================  parameter
"""
 hidden_layers,   [128,64]
 learning_rate,     [1e-3,1e-5,1e-7]  
 dropout=0,           
 activate_function='relu',     relu  sigmoid   tanh  LeakyReLU
 epoch=2000,
 batch_size=128,
 is_standard=False,
 weight_decay=1e-8,
 device=0,
 use_more_gpu=False,
 Dimensionality_reduction_method='None',
 save_path='ANN_Result'
 
 HL,LR1,LR2,LR3,EP,BS,DP,WD 
"""

hidden_layers_List = [ [128,64,8],[64,64,8],[256,128,16], [64,8], [32,4]]
activate_function_list = ["relu","sigmoid","tanh","LeakyReLU"]




class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType,save_path, X_train, y_train, X_test, y_test):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'catboost'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 9  # 初始化Dim（决策变量维数）
        #          HL, LR1, LR2, LR3, EP,BS, DP, WD, AF
        varTypes = [1,   0,   0,   0,  1, 1,  0,  0, 1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb =       [0, 1e-8, 1e-8, 1e-8,200,  8,  0, 0 ,0 ]  # 决策变量下界
        ub =       [len(hidden_layers_List)-1, 5e-2, 5e-2, 5e-2 ,10000 , 256 , 0.9, 0.9 ,len(activate_function_list)-1]  # 决策变量上界

        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 获取目标训练集和 测试集
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(len(self.X_train), len(self.y_train), len(self.X_test), len(self.y_test))
        # 获取保存路径
        self.save_path = save_path

        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
            # 使用多進程需要加個鎖 因為我還會保存結果
            self.lock = Manager().Lock()

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip( list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.X_train] * pop.sizes,
                [self.y_train] * pop.sizes,
                [self.X_test] * pop.sizes,
                [self.y_test] * pop.sizes,
                [self.lock] * pop.sizes,
                [self.save_path] * pop.sizes))
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())



    def test(self, HL,LR1,LR2,LR3,EP,BS,DP,WD,AF):  # 代入优化后的C、Gamma对测试集进行检验
        HL = int(HL)
        AF = int(AF)
        EP = int(EP)
        BS = int(BS)

        ann = ANN(hidden_layers=hidden_layers_List[HL],
                  learning_rate=[LR1, LR2, LR3],
                  epoch=EP,
                  batch_size=BS,
                  save_path=self.save_path,
                  dropout=DP,
                  weight_decay=WD,
                  activate_function=activate_function_list[AF],
                  is_standard="StandardScaler",
                  use_more_gpu = True,
                  Dimensionality_reduction_method='None')  # 采用梯度衰减策略

        ann.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        ann.score()

        ann.save()

        ObjV_i = ann.fitness()
        print("ObjV_i", ObjV_i)

        # loss_function_index = int(loss_function_index)
        # lossList = ["RMSE", "MultiRMSE", "MAE"]
        # parameterDict = {"learning_rate": learning_rate, "depth": depth, "loss_function": lossList[loss_function_index],"task_type":"GPU"}
        # CB_Regressor = cb.CatBoostRegressor(**parameterDict)
        # CB_Regressor.fit(self.X_train, self.y_train)
        # y_pre_CB = CB_Regressor.predict(self.X_test)
        # ObjV_i = reg_calculate(self.y_test, y_pre_CB)



def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    X_train = args[2]
    y_train = args[3]
    X_test = args[4]
    y_test = args[5]
    lock = args[6]
    save_path = args[7]

    HL,LR1,LR2,LR3,EP,BS,DP,WD,AF = Vars[i, 0],Vars[i, 1],Vars[i, 2],Vars[i, 3],Vars[i, 4],Vars[i, 5],Vars[i, 6],Vars[i, 7],Vars[i, 8]
    HL = int(HL)
    AF = int(AF)
    EP = int(EP)
    BS = int(BS)

    ann = ANN(hidden_layers=hidden_layers_List[HL],
              learning_rate=[LR1,LR2,LR3],
              epoch=EP,
              batch_size=BS,
              save_path=save_path,
              dropout=DP,
              weight_decay=WD,
              activate_function = activate_function_list[AF],
              is_standard="StandardScaler",
              use_more_gpu = True,
              Dimensionality_reduction_method='None')  # 采用梯度衰减策略

    ann.fit(X_train, y_train, X_test, y_test)
    ann.score()
    with lock:
        ann.save()

    ObjV_i = ann.fitness()


    # print("X_train {} y_train {} X_test {} y_test {}".format(np.array(X_train).shape,  np.array(y_train).shape, np.array(X_test).shape, np.array(y_test).shape))
    # print("subAimFunc {} times".format(i))
    # learning_rate = Vars[i, 0]
    # depth = Vars[i, 1]
    # loss_function_index = int(Vars[i, 2])
    #
    # lossList = ["RMSE", "MultiRMSE", "MAE"]
    # parameterDict = {"learning_rate": learning_rate, "depth": depth, "loss_function": lossList[loss_function_index],"task_type":"GPU"}
    # CB_Regressor = cb.CatBoostRegressor(**parameterDict)
    # CB_Regressor.fit(X_train, y_train)
    # y_pre_CB = CB_Regressor.predict(X_test)
    # ObjV_i = reg_calculate(y_test, y_pre_CB)
    # print("ObjV_i",ObjV_i)
    return [ObjV_i]

def mainFun(X_train, X_test, y_train, y_test,save_path,PoolType = 'Process'):
    """===============================实例化问题对象==========================="""
    # PoolType = 'Process'  # 设置采用多线程，若修改为: PoolType = 'Process'，则表示用多进程
    problem = MyProblem(PoolType, X_train, X_test, y_train, y_test)  # 生成问题对象
    """=================================种群设置=============================="""
    Encoding = 'RI'  # 编码方式
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 50  # 最大进化代数
    myAlgorithm.trappedValue = 1e-6  # “进化停滞”判断阈值
    myAlgorithm.maxTrappedCount = 10  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
    myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save()  # 把最优个体的信息保存到文件中
    """=================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为：%s' % (BestIndi.ObjV[0][0]))
        print('最优的控制变量值为：')
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])
        """=================================检验结果==============================="""
        testParameterDict = {"HL":BestIndi.Phen[0, 0],"LR1":BestIndi.Phen[0, 1],"LR2":BestIndi.Phen[0, 2],"LR3":BestIndi.Phen[0, 3],"EP":BestIndi.Phen[0, 4],
                             "BS":BestIndi.Phen[0, 5],"DP":BestIndi.Phen[0, 6],"WD":BestIndi.Phen[0, 7],"AF":BestIndi.Phen[0, 8]}
        problem.test(**testParameterDict)
    else:
        print('没找到可行解。')


if __name__ == '__main__':
    dataDict = train_test_split()
    for k, v in dataDict.items():
        print(k)
        X_train, x_test, y_train, y_test = v["X_train"], v["X_test"], v["y_train"], v["y_test"]
        print(len(X_train), len(x_test), len(y_train), len(y_test),
              os.path.join("./regression/", k, "ANN"))
        mainFun(X_train, x_test, y_train, y_test, os.path.join("./regression/", k, "ANN"))

