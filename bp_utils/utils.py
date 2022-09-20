# 定义一些 模型训练时用到的 函数
import math
import random

import numpy as np
def one_hot(arr):
    """
    对 数据标签 矩阵  进行 独热编码
    """
    w = len(arr)
    h = max(arr) + 1
    z = np.zeros([w, h])

    for i in range(w):
        j = int(arr[i])
        z[i][j] = 1
    return z

def data_normalize(dataset_x):
    """
    对训练集进行归一化处理
    x = (x-min)/(max-min)
    """
    ds_max = np.max(dataset_x,axis=0)
    ds_min = np.min(dataset_x,axis=0)

    dataset_x_pro = (dataset_x-ds_min)/(ds_max-ds_min)
    return dataset_x_pro




def train_test_split(dataset_x,datset_y,class_num,test_ratio=0.3,random_state = -1):
    """
    对 数据集进行划分，得到训练集 和 测试集
    Parameters
    ----------
    dataset_x,dateste_y：数据及其标签
    class_num：类别数目
    test_ration：测试集所占比例

    """
    ds_num,feature_num = dataset_x.shape[0],dataset_x.shape[1]

    class_item_index_list = []
    class_item_num_list = []
    class_train_test_item_num_list = []
    test_sum = 0
    for i in range(class_num):
        class_item_index = np.where(datset_y==i)[0]
        class_item_num = len(class_item_index)
        class_item_num_list.append(class_item_num)
        test_sum = test_sum + math.ceil(class_item_num * test_ratio)
        class_train_test_item_num_list.append([class_item_num- math.ceil(class_item_num * test_ratio) ,math.ceil(class_item_num * test_ratio)])
        class_item_index_list.append(class_item_index)


    train_x_ds ,test_x_ds= np.zeros([ds_num-test_sum,feature_num]),np.zeros([test_sum,feature_num])
    train_y_ds_list,test_y_ds_list = [],[]
    train_start_index ,test_start_index= 0,0
    for i,class_item_num in enumerate(class_item_num_list):
        class_item_index = class_item_index_list[i]
        np.random.shuffle(class_item_index)

        train_ds_item_num,test_ds_item_num = class_train_test_item_num_list[i][0],class_train_test_item_num_list[i][1]
        train_x_ds[train_start_index:train_start_index+train_ds_item_num,:]= dataset_x[class_item_index[:train_ds_item_num],:]
        test_x_ds[test_start_index:test_start_index+test_ds_item_num,:]= dataset_x[class_item_index[train_ds_item_num:],:]

        train_y_ds_list = train_y_ds_list+ [i]*(class_item_num - test_ds_item_num)
        test_y_ds_list = test_y_ds_list + [i]*test_ds_item_num

        train_start_index = train_start_index+train_ds_item_num
        test_start_index = test_start_index + test_ds_item_num

    train_y_ds,test_y_ds = np.array(train_y_ds_list), np.array(test_y_ds_list)
    # 对训练数据集 再进行一次乱序处理
    index = [i for i in range(ds_num-test_sum)]
    if random_state != -1:
        random.seed(random_state)
    random.shuffle(index)
    train_x_ds = train_x_ds[index]
    train_y_ds = train_y_ds[index]
    return train_x_ds,test_x_ds,train_y_ds,test_y_ds









