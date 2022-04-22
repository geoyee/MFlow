from ..core import NameScope, Variable
from ..lays import Linear
from typing import Dict, Union


def MultilayerPerception(
    input_size: int, classes: int, hidden_layers: int, activation: Union[str, None]
) -> Dict:
    """构造多分类逻辑回归模型的计算图

    Args:
        input_size (int): 输入维数，即特征数
        classes (int): 类别数
        hidden_layers (int): 数组，包含每个隐藏层的神经元数
        activation (str or None): 指定隐藏层激活函数类型，若为 None 则无激活函数

    Returns:
        Dict:
            x (Variable): 输入变量
            logits (Node): 多分类 logits
    """

    with NameScope("Input"):
        x = Variable((input_size, 1), init=False, trainable=False)
    with NameScope("Hidden"):
        output = x
        for size in hidden_layers:
            output = Linear(output, input_size, size, activation)
            input_size = size
    with NameScope("Logits"):
        logits = Linear(output, input_size, classes, None)  # 无激活函数
    return x, logits
