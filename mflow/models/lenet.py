from ..core import NameScope, Variable
from ..ops import Concat, SoftMax
from ..lays import Conv, Pooling, Linear
from typing import Tuple


# 太慢的跑起来
def LeNet() -> Tuple:
    """Lenet的keras实现"""
    with NameScope("Input"):
        x = Variable(size=(28, 28), trainable=False)
    with NameScope("Model"):
        conv1 = Conv([x], (28, 28), 32, (5, 5), "ReLU", padding="valid")  # 32x24x24
        poling1 = Pooling(conv1, (2, 2), (2, 2))  # 32x12x12
        conv2 = Conv(poling1, (12, 12), 64, (5, 5), "ReLU", padding="valid")  # 64x8x8
        poling2 = Pooling(conv2, (2, 2), (2, 2))  # 64x4x4
        fc1 = Linear(Concat(*poling2), 1024, 100, "ReLU")  # 100
        pred = Linear(fc1, 100, 10, None)  # 10
    with NameScope("SoftMax"):
        softmax = SoftMax(pred)
    return x, pred, softmax


# FOR TEST
def MnistNet() -> Tuple:
    with NameScope("Input"):
        x = Variable(size=(28, 28), trainable=False)
    with NameScope("Model"):
        conv1 = Conv([x], (28, 28), 3, (5, 5), "ReLU")  # 3x28x28
        poling1 = Pooling(conv1, (3, 3), (2, 2))  # 3x14x14
        conv2 = Conv(poling1, (14, 14), 3, (3, 3), "ReLU")  # 3x14x14
        poling2 = Pooling(conv2, (3, 3), (2, 2))  # 3x7x7
        fc1 = Linear(Concat(*poling2), 147, 120, "ReLU")  # 120
        pred = Linear(fc1, 120, 10, None)  # 10
    with NameScope("SoftMax"):
        softmax = SoftMax(pred)
    return x, pred, softmax
