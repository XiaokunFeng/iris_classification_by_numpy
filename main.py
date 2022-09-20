# 鸢尾花识别 主程序

import math
import numpy as np
import pandas as pd
from bp_utils.layers import FullyConnected,Softmax
from bp_utils.losses import CrossEntropy
from bp_utils.optimizers import SGD,MOBP
from bp_utils.utils import one_hot,data_normalize,train_test_split

from torch.utils.tensorboard import SummaryWriter

class BPNet(object):
    # 基于已有模块，定义BP网络
    def __init__(self,optimizer,hidden_num=4,init="std_normal"):
        self.hidden_num = hidden_num  # 单层隐藏层所含神经元个数
        self.optimizer = optimizer
        self.init = init
        self._init_parems()

    def _init_parems(self):
        self.fc1 = FullyConnected(
            4,self.hidden_num, act_fn_name="Tanh", optimizer=self.optimizer
        )
        self.fc2 = FullyConnected(
            self.hidden_num,3, optimizer=self.optimizer
        )
        self.softmax = Softmax(dim=1)

    def forward(self, X):
        X = self.fc1.forward(X)
        X = self.fc2.forward(X)
        X = self.softmax.forward(X)

        return X

    def backward(self,grad):
        out = self.fc2.backward(grad)
        out = self.fc1.backward(out)
        return out

    def update(self,cur_loss=None):
        self.fc1.update(para_layer_name='fc1',cur_loss=cur_loss)
        self.fc2.update(para_layer_name='fc2',cur_loss=cur_loss)



# ******************* 主程序入口 *******************
# 1.定义相关参数
experiment_intro = "main_function"      # 使用Tensorboard记录相关实验结果，便于后续分析
writer = SummaryWriter(comment=experiment_intro)
test_num = 50                           # 实验次数
epoch_num = 1000                        # 每次实验 训练的轮数
lr = 0.01                               # 学习率
optimizer = SGD(lr=0.01)                # 优化项类型
# optimizer = MOBP(lr=0.01,momentum=0.01)
hidden_num = 8                          # 单层神经元个数

# 2. 导入 IRIS 数据集
dataset = pd.read_csv('datasets/iris.csv')

dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2
dataset_x = dataset[dataset.columns[0:4]].values
dataset_y = dataset.species.values

# 对数据集进行归一化处理
dataset_x = data_normalize(dataset_x)

# 3.开始进行模型的训练和测试
for test_id in range(test_num):
    # 1）每次实验开始，随机进行一次训练集合测试集的划分
    train_X, test_X, train_y, test_y = train_test_split(dataset_x,dataset_y, class_num=3, test_ratio=0.3,random_state = 332)
    # 将标签数据 进行独热编码
    train_y0 = train_y.astype(np.int32)
    train_y = one_hot(train_y0)
    test_y0 = test_y.astype(np.int32)
    test_y = one_hot(test_y0)

    # 2）定义网络训练所用参数
    criterion = CrossEntropy() # 制定损失函数
    net = BPNet(hidden_num=hidden_num,optimizer=optimizer)
    batch_size = 35
    batch_size_num = math.ceil(train_X.shape[0]/batch_size)
    for epoch in range(epoch_num):
        all_index = list(range(0, train_X.shape[0] - 1))
        for batch_id in range(batch_size_num):
            start_index = batch_id*batch_size
            end_index = (batch_id+1)*batch_size   if (batch_id+1)*batch_size  <=train_X.shape[0] - 1 else train_X.shape[0] - 1
            train_data_index = all_index[start_index:end_index]
            train_X_temp = train_X[train_data_index, :]
            train_y_temp = train_y[train_data_index, :]
            out = net.forward(train_X_temp)
            loss = criterion(train_y_temp,out)
            dldy = criterion.grad(train_y_temp,out)
            dout = net.backward(dldy)
            net.update()

        y_out = np.argmax(out, 1)
        train_y0_temp = train_y0[train_data_index]
        acc_n = len(y_out)
        acc_num = np.sum(y_out == train_y0_temp)
        train_acc =  acc_num/ acc_n
        # 每20个epoch，就在测试集上测试一次
        if epoch % 20 == 0:
            writer.add_scalars(experiment_intro + '_train_loss', {str(test_id):loss}, epoch + 1)
            writer.add_scalars(experiment_intro + '_train_acc', {str(test_id):train_acc}, epoch + 1)
            print('number of epoch', epoch)
            predict_out = net.forward(test_X)
            loss = criterion(test_y, predict_out)
            predict_y = np.argmax(predict_out, 1)
            test_acc = np.sum(predict_y == test_y0) / len(predict_y)
            writer.add_scalars(experiment_intro + '_val_loss',  {str(test_id):loss}, epoch + 1)
            writer.add_scalars(experiment_intro + '_val_acc', {str(test_id):test_acc}, epoch + 1)

writer.close()
