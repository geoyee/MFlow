import numpy as np
from typing import Any
from ..core import Node


# 求e的x次方
def _ePower(x, trunc=1e2):
    # 数值截断，防止溢出
    return np.power(np.e, np.where(x > trunc, trunc, x)).astype("float32")


# 算子类
class Operator(Node):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Operator, self).__init__(*parents, **kwargs)