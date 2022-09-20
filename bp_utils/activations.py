# 将用到的激活函数 封装成类
import numpy as np
from abc import ABC,abstractmethod


class ActivationBase(ABC):
    """
    定义一个 激活函数类 基类
    -----
    包含了 前向计算 函数  fn()
    以及 一阶导数函数  grad()
    """
    def __init__(self):
        super(ActivationBase, self).__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        raise NotImplementedError

    @abstractmethod
    def grad(self,x):
        raise NotImplementedError

class Tanh(ActivationBase):
    """
    将tanh()函数封装成类
    -----
    包含了 前向计算 函数  fn()
    以及 一阶导数函数  grad()
    """
    def __init__(self):
        super(Tanh, self).__init__()
    def fn(self, z):
        return np.tanh(z)

    def grad(self, x):
        return 1 - np.tanh(x) ** 2

class ReLU(ActivationBase):
    """
    将relu()函数封装成类
    -----
    包含了 前向计算 函数  fn()
    以及 一阶导数函数  grad()
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int)

class Sigmoid(ActivationBase):
    """
    将sigmoid()函数封装成类
    -----
    包含了 前向计算 函数  fn()
    以及 一阶导数函数  grad()
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def fn(self, z):
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)


class Identity(ActivationBase):
    """
    将直接映射函数f(x)=x函数封装成类
    -----
    包含了 前向计算 函数  fn()
    以及 一阶导数函数  grad()
    """
    def __init__(self):
        super(Identity, self).__init__()

    def fn(self, z):
        return  z
    def grad(self, x):
        return np.ones_like(x)