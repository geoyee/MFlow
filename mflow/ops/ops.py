import numpy as np
from typing import Any
from ..core import Node


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
        self.value = np.mat(np.zeros_like(self.nparents[0]))
        for parent in self.nparents:
            self.value += parent.value

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.mat(np.eye(self.dim))


class MatMal(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(MatMal, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        assert len(self.nparents) == 2 and \
            self.nparents[0].shape[1] == self.nparents[1].shape[0]
        self.value = self.nparents[0].value * self.nparents[1].value

    def calcJacobi(self, parent: Any) -> np.matrix:
        zeros = np.mat(np.zeros((self.dim, parent.dim)))
        if parent is self.nparents[0]:
            return fillDiag(zeros, self.nparents[1].value.T)
        else:
            jacobi = fillDiag(zeros, self.nparents[0].value)
            row_sort = np.arange(self.dim).reshape(self.shape[::-1]).T.ravel()
            col_sort = np.arange(parent.dim).reshape(parent.shape[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Step(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Step, self).__init__(*parents, **kwargs)

    def calcValue(self) -> None:
        self.value = np.mat(np.where(self.nparents[0].value >= 0.0, 1.0, 0.0))

    def calcJacobi(self, parent: Any) -> np.matrix:
        # np.mat(np.eye(self.dim))  # FIXME:
        return np.zeros(np.where(self.nparents[0].value >= 0.0, 0.0, -1.0))