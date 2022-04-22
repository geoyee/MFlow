from ..core import NameScope, Variable
from ..ops import Concat, SoftMax
from ..lays import Conv, Pooling, Linear
from typing import Dict


def MNISTCNN(input_size: Dict = (28, 28), classes: int = 10) -> Dict:
    """_summary_

    Args:
        input_size (Dict, optional): 输入图像大小，默认为 (28, 28)
        classes (int, optional): 类别数，默认为 10

    Returns:
        Dict:
            x (Variable): 输入变量
            softmax (Node): 多分类 softmax
    """
    with NameScope("Input"):
        x = Variable(size=input_size, trainable=False)
    with NameScope("Model"):
        conv1 = Conv([x], input_size, 3, (5, 5), "ReLU")
        poling1 = Pooling(conv1, (3, 3), (2, 2))
        conv2 = Conv(poling1, (14, 14), 3, (3, 3), "ReLU")  # 28 / 2
        poling2 = Pooling(conv2, (3, 3), (2, 2))
        fc1 = Linear(Concat(*poling2), 147, 120, "ReLU")  # 3 * 7 * 7
        pred = Linear(fc1, 120, classes, None)
    with NameScope("SoftMax"):
        softmax = SoftMax(pred)
    return x, softmax
