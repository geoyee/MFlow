from ..core import Variable
from ..ops import (Add, MatMal, ReLU, Logistic)


# 全连接层
def Linear(input, input_size, size, act="ReLU"):
    weight = Variable((size, input_size), trainable=True)
    bias = Variable((size, 1), trainable=True)
    affine = Add(MatMal(weight, input), bias)
    if act == "ReLU":
        return ReLU(affine)
    elif act == "Logistic":
        return Logistic(affine)
    else:
        return affine