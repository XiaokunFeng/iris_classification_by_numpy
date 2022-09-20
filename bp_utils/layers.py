# 将用到的网络层 封装成类
from abc import ABC, abstractmethod
import numpy as np
from .initializers import init_weights
from .activations import Tanh,ReLU,Identity


class LayerBase(ABC):
    """
    关于网络层的抽象基础类
    """
    def __init__(self, optimizer=None):

        self.X = []
        self.act_fn = None
        self.optimizer = optimizer

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}

        super().__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        raise NotImplementedError

    def flush_gradients(self):
        """
        清空梯度信息
        """
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, para_layer_name='',cur_loss=None):
        """
        模型参数更新函数
        """
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v,para_layer_name+k, cur_loss)
        self.flush_gradients()

class FullyConnected(LayerBase):
    """
    将全连接层 封装成类
    """
    def __init__(self, n_in,n_out, act_fn_name="", init="std_uniform", optimizer=None):
        super().__init__(optimizer)

        self.n_in = n_in
        self.n_out = n_out
        self.act_fn_name = act_fn_name
        self.init = init
        self.parameters = {"W": None, "b": None}
        self._init_params()

    def _init_params(self):
        b = np.zeros((1, self.n_out))
        W = init_weights((self.n_in, self.n_out),act_fu_name = self.act_fn_name,mode="std_normal")

        self.parameters = {"W": W, "b": b}
        self.derived_variables = {"Z": []}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        if self.act_fn_name == "Relu":
            self.act_fn = ReLU()
        elif self.act_fn_name == "Tanh":
            self.act_fn = Tanh()
        elif self.act_fn_name == "":
            self.act_fn = Identity()



    def forward(self, X,retain_derived=True):
        """前向传播过程"""
        Y, Z = self._fwd(X)
        if retain_derived:
            self.X = []
            self.derived_variables["Z"] = []
            self.X.append(X)
            self.derived_variables["Z"].append(Z)

        return Y
    def _fwd(self, X):
        """具体的前向传播计算过程"""
        W = self.parameters["W"]
        b = self.parameters["b"]

        Z = X @ W + b
        Y = self.act_fn(Z)
        return Y, Z
    def backward(self, dLdy):
        """反向传播过程"""
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx, dw, db = self._bwd(dy, x)
            dX.append(dx)

            self.gradients["W"] += dw
            self.gradients["b"] += db

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """具体的反向传播计算过程"""
        W = self.parameters["W"]
        b = self.parameters["b"]

        Z = X @ W + b
        dZ = dLdy * self.act_fn.grad(Z)

        dX = dZ @ W.T
        dW = X.T @ dZ
        dB = dZ.sum(axis=0, keepdims=True)
        return dX, dW, dB

class Softmax(LayerBase):
    """
    将softmax激活函数，当成网络层来进行封装
    """
    def __init__(self, dim=-1, optimizer=None):

        super().__init__(optimizer)

        self.dim = dim
        self.n_in = None
        self.is_initialized = False

    def _init_params(self):
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}
        self.is_initialized = True

    def forward(self, X, retain_derived=True):
        """前向传播过程"""
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = self._fwd(X)

        self.X= X

        return Y

    def _fwd(self, X):
        """前向传播具体计算过程"""
        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))  # 整体除以（exp(X_max)）,避免出现溢出情况
        return e_X / e_X.sum(axis=self.dim, keepdims=True)

    def backward(self, dLdy, retain_grads=True):
        """反向传播过程"""

        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx = self._bwd(dy, x)
            dX.append(dx)

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """反向传播具体的计算流程"""
        dX = []
        for dy, x in zip(dLdy, X):
            dxi = []
            for dyi, xi in zip(*np.atleast_2d(dy, x)):
                yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1)
                dyidxi = np.diagflat(yi) - yi @ yi.T
                dxi.append(dyi @ dyidxi)
            dX.append(dxi)
        return np.array(dX).reshape(*X.shape)
