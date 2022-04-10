import numpy as np
from typing import Any
from .base import  Operator, _ePower
from .act import SoftMax


""" 注意
在每个算子类的计算重载后
请将数据的类型使用`astype("float32")`进行转换
可以避免numpy的一些错误
"""


# 损失函数类
class Loss(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Loss, self).__init__(*parents, **kwargs)
        self.eps = 1e-12


# 感知机损失
class PerceptionLoss(Loss):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(PerceptionLoss, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        self.value = np.mat(np.where(
            self.nparents[0].value >= 0.0, 0.0, -self.nparents[0].value)).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel()).astype("float32")


# 对数损失
class LogLoss(Loss):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(LogLoss, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        assert len(self.nparents) == 1
        x = self.nparents[0].value
        ix = -x
        self.value = np.log(1 + _ePower(ix)).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        x = parent.value
        diag = -1 / (1 + _ePower(x))
        return np.diag(diag.ravel()).astype("float32")


# 交叉熵损失
class CrossEntropyWithSoftMax(Loss):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(CrossEntropyWithSoftMax, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        self.prob = SoftMax.softmax(self.nparents[0].value).astype("float32")
        self.value = np.mat(-np.sum(np.multiply(
            self.nparents[1].value, np.log(self.prob + self.eps)))).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        # 交叉熵时计算Jacobi比SoftMax计算Jacobi来的更容易
        if parent is self.nparents[0]:
            return (self.prob - self.nparents[1].value).T.astype("float32")
        else:
            return (-np.log(self.prob)).T.astype("float32")