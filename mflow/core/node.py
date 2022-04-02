import numpy as np
from typing import Tuple, Any
from .graph import DefaultGraph


class Node(object):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        self.graph = kwargs.get("graph", DefaultGraph)
        self.saved = kwargs.get("saved", True)
        self._setName(**kwargs)
        self.value = None
        self.jacobi = None
        self.childrens = []
        self.parents = parents
        for parent in self.parents:
            parent.childrens.append(self)
        self.graph.addNode(self)

    @property
    def dim(self) -> int:
        return self.value[0] * self.value[1]

    @property
    def shape(self) -> Tuple:
        return self.value.shape

    def _setName(self, **kwargs: Any) -> None:
        self.name = kwargs.get(
            "name", 
            "{}:{}".format(self.__class__.__name__, self.graph.nodeConunt)
        )
        if self.graph.name_scope:
            self.name = "{}/{}".format(self.graph.name_scope, self.name)

    def forward(self) -> None:
        for parent in self.parents:
            parent.forward()
        self.calcValue()

    def calcValue(self) -> None:
        NotImplementedError()

    def resetValue(self, recursive: bool=True) -> None:
        self.value = None
        if recursive:
            for child in self.childrens:
                child.resetValue()

    def backward(self, result: Any) -> None:
        if self.jacobi is None and self is result:
            self.jacobi = np.mat(np.eye(self.dim))
        else:
            self.jacobi = np.mat(np.zeros(result.dim, self.dim))
            for child in self.childrens:
                if child.valus is not None:
                    self.jacobi += child.backward(result) * child.calcJacobi(self)
        return self.jacobi

    def calcJacobi(self, parent: Any) -> None:
        NotImplementedError()

    def clearJacobi(self) -> None:
        self.jacobi = None


class Variable(Node):
    def __init__(self, dim: Tuple[int], trainable: bool=True, **kwargs: Any) -> None:
        super(Variable, self).__init__(self, **kwargs)
        self.dim = dim
        self.trainable = trainable
        self.initValue()

    def initValue(self) -> None:
        self.value = np.mat(np.random.normal(0, 0.001, self.dim))

    def setValue(self, value: np.matrix) -> bool:
        if isinstance(value, np.matrix) and value.shape == self.dim:
            self.resetValue()
            self.value = value
            return True
        return False