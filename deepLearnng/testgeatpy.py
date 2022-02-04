# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from sklearn import datasets

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


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType):  # PoolType是取值为'Process'或'Thread'的字符串
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 2  # 初始化Dim（决策变量维数）
        varTypes = [0, 0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [2 ** (-8)] * Dim  # 决策变量下界
        ub = [2 ** 8] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 目标函数计算中用到的一些数据
        # fp = open('iris_train.data')
        # datas = []
        # data_targets = []
        #
        # for line in fp.readlines():
        #     line_data = line.strip('\n').split(',')
        #     data = []
        # for i in line_data[0:4]:
        #     data.append(float(i))
        # datas.append(data)
        # data_targets.append(line_data[4])
        #
        #
        # fp.close()

        iris = datasets.load_iris()
        datas = iris.data[:, :2]  # we only take the first two features.
        data_targets = iris.target

        self.data = preprocessing.scale(np.array(datas))  # 训练集的特征数据（归一化）
        self.dataTarget = np.array(data_targets)
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小


    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(zip(list(range(pop.sizes)), [Vars] * pop.sizes, [self.data] * pop.sizes, [
            self.dataTarget] * pop.sizes))  # pop.sizes就是种群的规模,args=[[种群大小],[决策变量*种群大小],[特征*种群大小],[分类target*种群大小]]，args含义见下面subAimFunc(args)函数


        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))  # 目标函数是一个np.array，也就是每次调参的分类结果的交叉验证得分
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())


    def test(self, C, G):  # 代入优化后的C、GasubAimFuncmma对测试集进行检验
        # 读取测试集数据
        fp = open('iris_test.data')
        datas = []
        data_targets = []


        for line in fp.readlines():
            line_data = line.strip('\n').split(',')
            data = []
        for i in line_data[0:4]:
            data.append(float(i))
        datas.append(data)
        data_targets.append(line_data[4])
        fp.close()
        data_test = preprocessing.scale(np.array(datas))  # 测试集的特征数据（归一化）
        dataTarget_test = np.array(data_targets)  # 测试集的标签数据
        svc = svm.SVC(C=C, kernel='rbf', gamma=G).fit(self.data, self.dataTarget)  # 创建分类器对象并用训练集的数据拟合分类器模型
        dataTarget_predict = svc.predict(data_test)  # 采用训练好的分类器对象对测试集数据进行预测
        print("测试集数据分类正确率 = %s%%" % (len(np.where(dataTarget_predict == dataTarget_test)[0]) / len(dataTarget_test) * 100))


def subAimFunc(args):  # 计算每一次模型调参的分类结果的交叉验证得分,模型一共训练了pop.sizes次，每一次的模型训练作为一个种群的样本
    i = args[0]
    Vars = args[1]
    data = args[2]
    dataTarget = args[3]
    C = Vars[i, 0]  # 调参1
    G = Vars[i, 1]  # 调参2
    svc = svm.SVC(C=C, kernel='rbf', gamma=G).fit(data, dataTarget)  # 创建分类器对象并用训练集的数据拟合分类器模型
    scores = cross_val_score(svc, data, dataTarget, cv=30)  # 计算交叉验证的得分
    ObjV_i = [scores.mean()]  # 把交叉验证的平均得分作为目标函数值


    return ObjV_i

# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea # import geatpy


if __name__ == '__main__':
    """===============================实例化问题对象==========================="""
    PoolType = 'Process' # 设置采用多进程，若修改为: PoolType = 'Thread'，则表示用多线程
    problem = MyProblem(PoolType) # 生成问题对象
    """=================================种群设置==============================="""
    Encoding = 'RI' # 编码方式
    NIND = 50 # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population) # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 30 # 最大进化代数
    myAlgorithm.trappedValue = 1e-6 # “进化停滞”判断阈值
    myAlgorithm.maxTrappedCount = 10 # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
    """==========================调用算法模板进行种群进化======================="""
    [population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板
    population.save() # 把最后一代种群的信息保存到文件中
    problem.pool.close() # 及时关闭问题类中的池，否则在采用多进程运算后内存得不到释放
    # 输出结果
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
    best_ObjV = obj_trace[best_gen, 1]
    print('最优的目标函数值为：%s'%(best_ObjV))
    print('最优的控制变量值为：')
    for i in range(var_trace.shape[1]):
        print(var_trace[best_gen, i])
    print('有效进化代数：%s'%(obj_trace.shape[0]))
    print('最优的一代是第 %s 代'%(best_gen + 1))
    print('评价次数：%s'%(myAlgorithm.evalsNum))
    print('时间已过 %s 秒'%(myAlgorithm.passTime))
    """=================================检验结果==============================="""
    # problem.test(C = var_trace[best_gen, 0], G = var_trace[best_gen, 1])