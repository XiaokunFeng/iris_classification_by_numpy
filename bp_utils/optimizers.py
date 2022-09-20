# 将用到的优化器函数 封装成类
from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import norm
class OptimizerBase(ABC):
    def __init__(self, lr):
        self.cache = {}
        self.cur_step = 0
        self.hyperparameters = {}
        self.lr = lr

    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        """每次优化步数都加 1"""
        self.cur_step += 1

    def reset_step(self):
        """初始化步数计数器为0"""
        self.cur_step = 0

    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss=None):
        raise NotImplementedError


class SGD(OptimizerBase):
    """
    将随机梯度下降策略封装成类
    """
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def update(self, param, param_grad, param_name, cur_loss=None):
        C = self.cache

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # 避免梯度过大
        if norm(param_grad) > np.inf:
            param_grad = param_grad * np.inf / norm(param_grad)

        update = self.lr * param_grad
        self.cache[param_name] = update
        return param - update

class MOBP(OptimizerBase):
    """
   将动量法策略封装成类
   """
    def __init__(self, lr=0.01,momentum=0.1):
        super().__init__(lr)
        self.lr = lr*(1-momentum)
        self.momentum =lr*momentum
    def update(self, param, param_grad, param_name, cur_loss=None):
        C = self.cache

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # 避免梯度过大
        if norm(param_grad) > np.inf:
            param_grad = param_grad * np.inf / norm(param_grad)

        update = self.momentum * C[param_name] + self.lr * param_grad
        self.cache[param_name] = update
        return param - update
