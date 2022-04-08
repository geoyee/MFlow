import numpy as np
from typing import Any
from .base import Operator, _ePower


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
        self.value = np.mat(1 / (1 + _ePower(-x))).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1).astype("float32")


class SoftMax(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(SoftMax, self).__init__(*parents, **kwargs)

    @staticmethod
    def softmax(x: np.matrix) -> np.matrix:
        ep = _ePower(x)
        return np.mat(ep / sum(ep))

    def calcValue(self) -> None:
        self.value = SoftMax.softmax(self.nparents[0].value).astype("float32")

    # 不实现SoftMax的calcJocabi方法，训练时使用CrossEntropyWithSoftMax


class Tanh(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Tanh, self).__init__(*parents, **kwargs)
        
    def calcValue(self) -> None:
        x = self.nparents[0].value
        # FIXME: 还是可能溢出，不知为何
        self.value = np.mat(
            (_ePower(x) - _ePower(-x)) / (_ePower(x) + _ePower(-x))).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.mat(1 - np.power(self.value, 2)).astype("float32")


class ReLU(Operator):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(ReLU, self).__init__(*parents, **kwargs)
        self.nslope = 0.1  # LeakyReLU:0.1, ReLU:0

    def calcValue(self) -> None:
        self.value = np.mat(np.where(
            self.nparents[0].value > 0.0,
            self.nparents[0].value,
            self.nslope * self.nparents[0].value)).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.diag(np.where(
            self.nparents[0].value.A1 > 0.0, 1.0, self.nslope)).astype("float32")