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
            self.nparents[0].value >= 0.0, 0.0, -self.nparents[0].value))

    def calcJacobi(self, parent: Any) -> np.matrix:
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())