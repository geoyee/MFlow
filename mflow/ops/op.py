import numpy as np
from typing import Any
from .base import Operator
from ..core import Node


""" 注意
在每个算子类的计算重载后
请将数据的类型使用`astype("float32")`进行转换
可以避免numpy的一些错误
"""


# 将filler矩阵填充在to_be_filled的对角线上
def fillDiag(to_be_filled: np.matrix, filler: np.matrix) -> np.matrix:
    assert (
        to_be_filled.shape[0] / filler.shape[0]
        == to_be_filled.shape[1] / filler.shape[1]
    )
    n = int(to_be_filled.shape[0] / filler.shape[0])
    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r : (i + 1) * r, i * c : (i + 1) * c] = filler
    return to_be_filled


class Add(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Add, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        self.value = np.mat(np.zeros(self.nparents[0].shape))
        for parent in self.nparents:
            self.value += parent.value

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.mat(np.eye(self.dim)).astype("float32")


# 矩阵乘法
class MatMal(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(MatMal, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        assert (
            len(self.nparents) == 2
            and self.nparents[0].shape[1] == self.nparents[1].shape[0]
        )
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
        self.value = np.multiply(self.nparents[0].value, self.nparents[1].value).astype(
            "float32"
        )

    def calcJacobi(self, parent: Any) -> np.matrix:
        if parent is self.nparents[0]:
            return np.diag(self.nparents[1].value.A1).astype("float32")
        else:
            return np.diag(self.nparents[0].value.A1).astype("float32")


class Reshape(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Reshape, self).__init__(*parents, **kwargs)
        self.new_shape = kwargs.get("shape")
        assert isinstance(self.new_shape, tuple) and len(self.new_shape) == 2

    def calcValue(self) -> None:
        self.value = self.nparents[0].value.reshape(self.new_shape)

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent is self.nparents[0]
        return np.mat(np.eye(self.dim)).astype("float32")


class Concat(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Concat, self).__init__(*parents, **kwargs)
        self.axis = kwargs.get("axis", 1)

    def calcValue(self) -> None:
        self.value = np.concatenate(
            [p.value.flatten() for p in self.nparents], axis=self.axis
        ).T

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent in self.nparents
        dims = [p.dim for p in self.nparents]
        index = self.nparents.index(parent)
        dim = int(parent.dim)
        assert dim == dims[index]
        jacobi = np.mat(np.zeros((self.dim, dim)))
        start_row = int(np.sum(dims[:index]))
        jacobi[start_row : start_row + dim, 0:dim] = np.eye(dim)
        return jacobi.astype("float32")


# 焊接点
class Welding(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Welding, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        assert len(self.nparents) == 1 and self.nparents[0] is not None
        self.value = self.nparents[0].value

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent is self.nparents[0]
        return np.mat(np.eye(self.dim)).astype("float32")

    def weld(self, node: Node) -> None:
        # 与之前的父节点断开
        if len(self.nparents) == 1 and self.nparents[0] is not None:
            self.nparents[0].nchildrens.remove(self)
        self.nparents.clear()
        # 与传入节点焊接
        self.nparents.append(node)
        node.nchildrens.append(self)
