import numpy as np
from ..core import Node
from typing import Any


class Metric(Node):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Metric, self).__init__(*parents, **kargs)
        kargs["saved"] = kargs.get("saved", False)
        self.init()

    def init(self) -> None:
        raise NotImplementedError()

    def reset(self) -> None:
        self.resetValue()
        self.init()

    def valueStr(self) -> str:
        return "{}: {:.4f}".format(self.__class__.__name__, self.value)

    @staticmethod
    def prob2Label(prob: np.ndarray, threshold: float = 0.5) -> int:
        if prob.shape[0] > 1:
            # 如果是多分类，预测类别为概率最大的类别
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            # 否则以thresholds为概率阈值判断类别
            labels = np.where(prob < threshold, -1, 1)
        return labels
