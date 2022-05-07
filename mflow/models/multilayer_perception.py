from ..core import NameScope, Variable
from ..lays import Linear
from typing import Tuple, List, Optional


def MultilayerPerception(
    input_size: int, classes: int, hidden_layers: List[int], activation: Optional[str]
) -> Tuple:
    """构造多分类逻辑回归模型的计算图

    Args:
        input_size (int): 输入维数，即特征数
        classes (int): 类别数
        hidden_layers (List[int]): 数组，包含每个隐藏层的神经元数
        activation (str or None): 指定隐藏层激活函数类型，若为 None 则无激活函数

    Returns:
        Tuple:
            x (Variable): 输入变量
            logits (Node): 多分类 logits
    """

    with NameScope("Input"):
        x = Variable((input_size, 1), init=False, trainable=False)
    with NameScope("Hidden"):
        output = Linear(x, input_size, hidden_layers[0], activation)
        input_size = hidden_layers[0]
        for i in range(1, len(hidden_layers)):
            output = Linear(output, input_size, hidden_layers[i], activation)
            input_size = hidden_layers[i]
    with NameScope("Logits"):
        logits = Linear(output, input_size, classes, None)  # 无激活函数
    return x, logits
