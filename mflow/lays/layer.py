from typing import List, Tuple, Union
from ..core import Variable, Node
from ..ops import *


# 全连接层
def Linear(
    input: Node, input_size: int, size: int, act: Union[str, None] = "ReLU"
) -> Operator:
    weight = Variable((size, input_size), trainable=True)
    bias = Variable((size, 1), trainable=True)
    affine = Add(MatMul(weight, input), bias)
    if act == "ReLU":
        return ReLU(affine)
    elif act == "Logistic":
        return Logistic(affine)
    elif act == "Tanh":
        return Tanh(affine)
    else:
        return affine


# 卷积层
def Conv(
    feat_maps: List,
    input_shape: Tuple,
    kernels: int,
    kernel_shape: Tuple,
    act: Union[str, None] = "ReLU",
) -> List:
    ones = Variable(input_shape, trainable=False)
    ones.setValue(np.mat(np.ones(input_shape)))
    outputs = []
    for _ in range(kernels):
        channels = []
        for fm in feat_maps:
            kernel = Variable(kernel_shape, trainable=True)
            channels.append(Convolve(fm, kernel))
        channels = Add(*channels)
        bias = ScalarMultiply(Variable(size=(1, 1), trainable=True), ones)
        affine = Add(channels, bias)
        if act == "ReLU":
            outputs.append(ReLU(affine))
        elif act == "Logistic":
            outputs.append(Logistic(affine))
        elif act == "Tanh":
            outputs.append(Tanh(affine))
        else:
            outputs.append(affine)
    assert len(outputs) == kernels
    return outputs


# 池化层
def Pooling(
    feat_maps: List, kernel_shape: Tuple, stride: Tuple, mode: str = "Max"
) -> List:
    outputs = []
    pooling = MaxPooling if mode == "Max" else AvePooling
    for fm in feat_maps:
        outputs.append(pooling(fm, size=kernel_shape, stride=stride))
    return outputs
