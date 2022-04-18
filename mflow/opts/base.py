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

    def applyGrads(
        self,
        node_grids_dict: Dict,
        summarize: bool = False,
        acc_idx: Union[None, int] = None,
    ) -> None:
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

    def update(self, var_grads: Union[None, Dict] = None) -> None:
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
