import numpy as np
from typing import Any
from ..core import Node


# 损失函数类
class Loss(Node):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Loss, self).__init__(*parents, **kwargs)


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
        # 数值戒断，防止溢出
        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        x = parent.value
        diag = -1.0 / (1.0 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(diag.ravel()).astype("float32")