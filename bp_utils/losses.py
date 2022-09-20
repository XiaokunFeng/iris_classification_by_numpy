# 将用到的 损失函数 封装成类
from abc import ABC, abstractmethod
import numpy as np

class LossBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        pass

class CrossEntropy(LossBase):
    """
    将交叉熵损失函数封装成类
    """
    def __init__(self):
        super().__init__()

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)
    @staticmethod
    def loss(y, y_pred):
        # loss的计算
        eps = np.finfo(float).eps
        cross_entropy = -np.sum(y * np.log(y_pred + eps))
        return cross_entropy
    @staticmethod
    def grad(y, y_pred):
        """loss关于y_pred的偏导数"""
        grad = y_pred - y
        return grad