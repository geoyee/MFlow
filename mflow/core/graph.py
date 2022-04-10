from typing import Any


# 计算图类
class Graph(object):
    def __init__(self) -> None:
        self.nodes = []
        self.name_scope = None

    def addNode(self, node: Any) -> None:
        self.nodes.append(node)

    def clearAllJacobis(self) -> None:
        for node in self.nodes:
            node.clearJacobi()

    def resetAllValues(self) -> None:
        for node in self.nodes:
            node.resetValue()

    @property
    def nodeConunt(self) -> int:
        return len(self.nodes)


# 默认全局计算图
DefaultGraph = Graph()
