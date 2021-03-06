import numpy as np
from typing import Any
from .base import Operator, _ePower


class Step(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Step, self).__init__(*parents, **kargs)

    def calcValue(self) -> None:
        self.value = np.mat(np.where(self.nparents[0].value >= 0.0, 1.0, 0.0)).astype(
            "float32"
        )

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.mat(np.zeros(self.dim)).astype("float32")


class Logistic(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Logistic, self).__init__(*parents, **kargs)

    def calcValue(self) -> None:
        x = self.nparents[0].value
        ix = -x
        self.value = np.mat(1.0 / (1.0 + _ePower(ix))).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        if self.value is None:
            raise ValueError("`self.value` is None.")
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1).astype(
            "float32"
        )


class SoftMax(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(SoftMax, self).__init__(*parents, **kargs)

    @staticmethod
    def softmax(x: np.matrix, eps: float = 1e-12) -> np.matrix:
        ep = _ePower(x)
        return np.mat(ep / (sum(ep) + eps))

    def calcValue(self) -> None:
        self.value = SoftMax.softmax(self.nparents[0].value).astype("float32")

    # 不实现SoftMax的calcJocabi方法，训练时使用CrossEntropyWithSoftMax


class Tanh(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Tanh, self).__init__(*parents, **kargs)

    def calcValue(self) -> None:
        x = self.nparents[0].value
        ix = -x
        self.value = np.mat(
            (_ePower(x) - _ePower(ix)) / (_ePower(x) + _ePower(ix) + self.eps)
        ).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.mat(1 - np.power(self.value, 2)).astype("float32")


class ReLU(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(ReLU, self).__init__(*parents, **kargs)
        self.nslope = 0.1  # LeakyReLU:0.1, ReLU:0

    def calcValue(self) -> None:
        self.value = np.mat(
            np.where(
                self.nparents[0].value > 0.0,
                self.nparents[0].value,
                self.nslope * self.nparents[0].value,
            )
        ).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.diag(
            np.where(self.nparents[0].value.A1 > 0.0, 1.0, self.nslope)
        ).astype("float32")
