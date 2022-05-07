from ..core import NameScope, Variable
from ..ops import Add, MatMul
from typing import Tuple


def LogisticRegression(input_size: int, classes: int) -> Tuple:
    """构造多分类逻辑回归模型的计算图

    Args:
        input_size (int): 输入维数，即特征数
        classes (int): 类别数

    Returns:
        Tuple:
            x (Variable): 输入变量
            logits (Node): 多分类 logits
    """

    with NameScope("Input"):
        x = Variable((input_size, 1), trainable=False)
    with NameScope("Parameter"):
        w = Variable((classes, input_size), trainable=True)
        b = Variable((classes, 1), trainable=True)
    with NameScope("Logits"):
        logits = Add(MatMul(w, x), b)
    return x, logits
