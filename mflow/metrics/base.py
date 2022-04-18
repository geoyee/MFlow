import numpy as np
from ..core import Node
from typing import Any


class Metric(Node):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Metric, self).__init__(*parents, **kwargs)
        kwargs["saved"] = kwargs.get("saved", False)
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
            labels = np.argmax(prob, axis=0)
        else:
            labels - np.where(prob < threshold, 0, 1)
        return labels
