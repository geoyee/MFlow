import numpy as np
from typing import Tuple, Any
from .graph import DefaultGraph


# 节点类
class Node(object):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        self.kargs = kargs
        self.graph = kargs.get("graph", DefaultGraph)
        self.saved = kargs.get("saved", True)
        self._setName(**kargs)
        self.value = None
        self.jacobi = None
        self.nchildrens = []
        self.nparents = list(parents)
        for parent in self.nparents:
            parent.nchildrens.append(self)
        self.graph.addNode(self)

    @property
    def dim(self) -> int:
        return self.value.shape[0] * self.value.shape[1]

    @property
    def shape(self) -> Tuple:
        return self.value.shape

    def _setName(self, **kargs: Any) -> None:
        self.name = kargs.get(
            "name", "{}:{}".format(self.__class__.__name__, self.graph.nodeConunt)
        )
        if self.graph.name_scope:
            self.name = "{}/{}".format(self.graph.name_scope, self.name)

    def forward(self) -> np.matrix:
        for parent in self.nparents:
            parent.forward()
        self.calcValue()
        return self.value

    def calcValue(self) -> None:
        NotImplementedError()

    def resetValue(self, recursive: bool = True) -> None:
        self.value = None
        if recursive:
            for child in self.nchildrens:
                child.resetValue()

    def backward(self, result: Any) -> np.matrix:
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dim))
            else:
                self.jacobi = np.mat(np.zeros((result.dim, self.dim)))
                for child in self.nchildrens:
                    if child.value is not None:
                        self.jacobi += child.backward(result) * child.calcJacobi(self)
        return self.jacobi

    def calcJacobi(self, parent: Any) -> np.matrix:
        NotImplementedError()

    def clearJacobi(self) -> None:
        self.jacobi = None


# 参数类
class Variable(Node):
    def __init__(self, size: Tuple, trainable: bool = True, **kargs: Any) -> None:
        super(Variable, self).__init__(**kargs)  # 变量没有父节点
        self.size = size
        self.trainable = trainable
        if trainable:
            self.initValue()

    def initValue(self) -> None:
        self.value = np.mat(np.random.normal(0, 0.001, self.size))

    def setValue(self, value: np.matrix) -> bool:
        if isinstance(value, np.matrix) and value.shape == self.size:
            self.resetValue()
            self.value = value
            return True
        return False

    def step(self, lr: float) -> None:
        self.setValue(self.value - lr * self.jacobi.T.reshape(self.shape))
