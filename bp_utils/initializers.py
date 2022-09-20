# 将用到的 参数初始化函数 进行实现
import numpy as np
#######################################################################
#                        Weight Initialization                        #
#######################################################################

def init_weights(weight_shape,act_fu_name,mode="glorot_uniform"):
    if mode not in [
        "std_normal"
    ]:
        raise ValueError("Unrecognize initialization mode: {}".format(mode))

    if  mode == "std_normal":
        initialized_weight = std_normal(weight_shape)
    return initialized_weight

def std_normal(weight_shape):
    return np.random.rand(weight_shape[0],weight_shape[1])
