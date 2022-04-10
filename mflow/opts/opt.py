import numpy as np
from typing import Dict, Union
from ..core import Graph, Node, getNodeByName, getTrainabledNode


# 优化器类
class Optimizer(object):
    def __init__(self, graph: Graph, target: Node, lr=0.01) -> None:
        self.graph = graph
        self.target = target
        self.lr = lr
        self.eps = 1e-12
        # Mini Batch中的样本梯度
        self.acc_grads = dict()
        self.acc_idx = 0

    def step(self) -> None:
        self.forwardAndBackward()
        self.acc_idx += 1

    def getGrad(self, node: Node):
        if node not in self.acc_grads.keys():
            raise KeyError("There are not node named {0}.".format(node.name))
        return self.acc_grads[node] / self.acc_idx

    def _update(self) -> None:
        NotImplementedError()

    def applyGrads(self, 
                   node_grids_dict: Dict, 
                   summarize: bool=False, 
                   acc_idx: Union[None, int]=None) -> None:
        for node, grid in node_grids_dict.items():
            if isinstance(node, Node):
                pass
            else:
                # 下面node可能是node_name
                target_node = getNodeByName(node)
                assert target_node is not None
                assert self.acc_grads[target_node].shape == grid.shape
                if summarize:
                    self.acc_grads[target_node] += grid
                else:
                    self.acc_grads[target_node] = grid
        if summarize:
            self.acc_idx += acc_idx
        else:
            if acc_idx is None:
                # 避免梯度更新时重复平均
                self.acc_idx = 1
            else:
                self.acc_idx = acc_idx

    def update(self, var_grads: Union[None, Dict]=None) -> None:
        if var_grads is not None:
            self.applyGrads(var_grads)
        # 更新
        self._update()
        # 清理
        self.acc_grads.clear()
        self.acc_idx = 0

    def forwardAndBackward(self) -> None:
        self.graph.clearAllJacobis()
        self.target.forward()
        # 参数进行反向传播
        var_nodes = getTrainabledNode(self.graph)
        for var_node in var_nodes:
            var_node.backward(self.target)
            grid = var_node.jacobi.T.reshape(var_node.shape)
            if var_node not in self.acc_grads:
                self.acc_grads[var_node] = grid
            else:
                self.acc_grads[var_node] += grid


# 梯度下降优化器
class GradientDescent(Optimizer):
    def __init__(self, graph: Graph, target: Node, lr: float=0.01) -> None:
        super(GradientDescent, self).__init__(graph, target, lr)

    def _update(self) -> None:
        var_nodes = getTrainabledNode(self.graph)
        for var_node in var_nodes:
            grad = self.getGrad(var_node)
            var_node.setValue(var_node.value - self.lr * grad)


class Momentum(Optimizer):
    def __init__(self, 
                 graph: Graph, 
                 target: Node, 
                 lr: float=0.01, 
                 momentum: float=0.9) -> None:
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
                self.hist_v[var_node] = self.momentum * self.hist_v[var_node] - \
                                        self.lr * grad
            var_node.setValue(var_node.value + self.hist_v[var_node])


class AdaGrad(Optimizer):
    def __init__(self, graph: Graph, target: Node, lr: float=0.01) -> None:
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
                var_node.value - self.lr * grad / 
                (np.sqrt(self.hist_s[var_node] + self.eps)))


class RMSProp(Optimizer):
    def __init__(self, 
                 graph: Graph, 
                 target: Node, 
                 lr: float=0.01, 
                 beta: float=0.9) -> None:
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
                self.hist_s[var_node] = self.beta * self.hist_s[var_node] + \
                                        (1 - self.beta) * np.power(grad, 2)
            var_node.setValue(
                var_node.value - self.lr * grad / 
                (np.sqrt(self.hist_s[var_node] + self.eps)))


class Adam(Optimizer):
    def __init__(self, 
                 graph: Graph, 
                 target: Node, 
                 lr: float=0.01, 
                 beta_1: float=0.9, 
                 beta_2: float=0.99) -> None:
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
                self.hist_v[var_node] = self.beta_1 * self.hist_v[var_node] + \
                                        (1 - self.beta_1) * grad
                self.hist_s[var_node] = self.beta_2 * self.hist_s[var_node] + \
                                        (1 - self.beta_2) * np.power(grad, 2)
            var_node.setValue(
                var_node.value - self.lr * self.hist_v[var_node] / 
                (np.sqrt(self.hist_s[var_node] + self.eps)))