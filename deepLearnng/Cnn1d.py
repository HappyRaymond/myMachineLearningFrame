from sklearn import datasets  # 导入库

boston = datasets.load_boston()  # 导入波士顿房价数据
from sklearn.model_selection import train_test_split
# check data shape
print("boston.data.shape %s , boston.target.shape %s"%(boston.data.shape,boston.target.shape))

from Frame import DeepLearningModel
from Frame import DeepLearningRegression

train = boston.data  # sample
target = boston.target  # target
# 切割数据样本集合测试集
X_train, x_test, y_train, y_true = train_test_split(train, target, test_size=0.2)  # 20%测试集；80%训练集


if __name__ == '__main__':
    cnn1d = DeepLearningModel.CNN1D(channel_num_list=[4,8,8,16], kernel_size=3, stride=1, padding=2, Pool_kernel_size=1,
                 dropout=0.4,activate_function = "relu", featureNum = 13)

    print(cnn1d)
    GRT = DeepLearningRegression.General_Regression_Training(cnn1d, learning_rate=[1e-2, 1e-5, 1e-8], epoch=2000,
                                                             use_more_gpu=False, batch_size=512,save_path='regression/TEST/CNN_1D')
    GRT.fit(X_train, y_train, x_test, y_true)
    GRT.save_results()

