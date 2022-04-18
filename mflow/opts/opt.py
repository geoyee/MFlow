import numpy as np
from .base import Optimizer
from ..core import Graph, Node, getTrainabledNode


# 梯度下降优化器
class GradientDescent(Optimizer):
    def __init__(self, graph: Graph, target: Node, lr: float = 0.01) -> None:
        super(GradientDescent, self).__init__(graph, target, lr)

    def _update(self) -> None:
        var_nodes = getTrainabledNode(self.graph)
        for var_node in var_nodes:
            grad = self.getGrad(var_node)
            var_node.setValue(var_node.value - self.lr * grad)


class Momentum(Optimizer):
    def __init__(
        self, graph: Graph, target: Node, lr: float = 0.01, momentum: float = 0.9
    ) -> None:
        super(Momentum, self).__init__(graph, target, lr)
        self.momentum = momentum
        self.hist_v = {}  # 累计历史速度

    def _update(self) -> None:
        var_nodes = getTrainabledNode(self.graph)
        for var_node in var_nodes:
            grad = self.getGrad(var_node)
            # 加入历史记录
            if var_node not in self.hist_v:
                self.hist_v[var_node] = -self.lr * grad
            else:
                self.hist_v[var_node] = (
                    self.momentum * self.hist_v[var_node] - self.lr * grad
                )
            var_node.setValue(var_node.value + self.hist_v[var_node])


class AdaGrad(Optimizer):
    def __init__(self, graph: Graph, target: Node, lr: float = 0.01) -> None:
        super(AdaGrad, self).__init__(graph, target, lr)
        self.hist_s = {}

    def _update(self) -> None:
        var_nodes = getTrainabledNode(self.graph)
        for var_node in var_nodes:
            grad = self.getGrad(var_node)
            # 累计梯度分量的平方和
            if var_node not in self.hist_s:
                self.hist_s[var_node] = np.power(grad, 2)
            else:
                self.hist_s[var_node] += np.power(grad, 2)
            var_node.setValue(
                var_node.value
                - self.lr * grad / (np.sqrt(self.hist_s[var_node] + self.eps))
            )


class RMSProp(Optimizer):
    def __init__(
        self, graph: Graph, target: Node, lr: float = 0.01, beta: float = 0.9
    ) -> None:
        super(RMSProp, self).__init__(graph, target, lr)
        self.beta = beta
        self.hist_s = {}

    def _update(self) -> None:
        var_nodes = getTrainabledNode(self.graph)
        for var_node in var_nodes:
            grad = self.getGrad(var_node)
            # 累计梯度分量的平方和
            if var_node not in self.hist_s:
                self.hist_s[var_node] = np.power(grad, 2)
            else:
                self.hist_s[var_node] = self.beta * self.hist_s[var_node] + (
                    1 - self.beta
                ) * np.power(grad, 2)
            var_node.setValue(
                var_node.value
                - self.lr * grad / (np.sqrt(self.hist_s[var_node] + self.eps))
            )


class Adam(Optimizer):
    def __init__(
        self,
        graph: Graph,
        target: Node,
        lr: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
    ) -> None:
        super(Adam, self).__init__(graph, target, lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.hist_v = {}
        self.hist_s = {}

    def _update(self) -> None:
        var_nodes = getTrainabledNode(self.graph)
        for var_node in var_nodes:
            grad = self.getGrad(var_node)
            # 累计梯度分量的平方和
            if var_node not in self.hist_v:
                self.hist_v[var_node] = grad
                self.hist_s[var_node] = np.power(grad, 2)
            else:
                self.hist_v[var_node] = (
                    self.beta_1 * self.hist_v[var_node] + (1 - self.beta_1) * grad
                )
                self.hist_s[var_node] = self.beta_2 * self.hist_s[var_node] + (
                    1 - self.beta_2
                ) * np.power(grad, 2)
            var_node.setValue(
                var_node.value
                - self.lr
                * self.hist_v[var_node]
                / (np.sqrt(self.hist_s[var_node] + self.eps))
            )
