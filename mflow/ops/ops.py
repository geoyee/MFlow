import numpy as np
from typing import Any
from ..core import Node


""" 注意
在每个算子类的计算重载后
请将数据的类型使用`astype("float32")`进行转换
可以避免numpy的一些错误
"""


# 算子类
class Operator(Node):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Operator, self).__init__(*parents, **kwargs)


# 将filler矩阵填充在to_be_filled的对角线上
def fillDiag(to_be_filled: np.matrix, filler: np.matrix) -> np.matrix:
    assert to_be_filled.shape[0] / filler.shape[0] == \
           to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])
    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler
    return to_be_filled


class Add(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Add, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        self.value = np.mat(np.zeros(self.nparents[0].shape))
        for parent in self.nparents:
            self.value += parent.value
        self.value = self.value.astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.mat(np.eye(self.dim)).astype("float32")


# 矩阵乘法
class MatMal(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(MatMal, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        assert len(self.nparents) == 2 and \
            self.nparents[0].shape[1] == self.nparents[1].shape[0]
        self.value = (self.nparents[0].value * self.nparents[1].value).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        zeros = np.mat(np.zeros((self.dim, parent.dim)))
        if parent is self.nparents[0]:
            return fillDiag(zeros, self.nparents[1].value.T)
        else:
            jacobi = fillDiag(zeros, self.nparents[0].value)
            row_sort = np.arange(self.dim).reshape(self.shape[::-1]).T.ravel()
            col_sort = np.arange(parent.dim).reshape(parent.shape[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort].astype("float32")


# 对应位置的元素相乘
class Multiply(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Multiply, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        self.value = np.multiply(
            self.nparents[0].value, self.nparents[1].value).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        if parent is self.nparents[0]:
            return np.diag(self.nparents[1].value.A1).astype("float32")
        else:
            return np.diag(self.nparents[0].value.A1).astype("float32")


class Step(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Step, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        self.value = np.mat(
            np.where(self.nparents[0].value >= 0.0, 1.0, 0.0)).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.mat(np.zeros(self.dim)).astype("float32")


class Logistic(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Logistic, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        x = self.nparents[0].value
        # 数值截断，防止溢出
        self.value = np.mat(
            1 / (1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1).astype("float32")


class SoftMax(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(SoftMax, self).__init__(*parents, **kwargs)

    @staticmethod
    def softmax(x: np.matrix) -> np.matrix:
        x[x > 1e2] = 1e2  # 数值截断，防止溢出
        ep = np.power(np.e, x)
        return ep / sum(ep)

    def calcValue(self) -> None:
        self.value = SoftMax.softmax(self.nparents[0].value)

    # 不实现SoftMax的calcJocabi方法，训练时使用CrossEntropyWithSoftMax