import numpy as np
from typing import Union, Any, List
from .node import Variable, Node
from .graph import DefaultGraph, Graph


def getNodeByName(
    node_name: str,
    name_scope: Union[None, str] = None,
    graph: Union[None, Graph] = None,
) -> Union[None, Node]:
    graph = DefaultGraph if graph is None else graph
    node_name = (name_scope + "/" + node_name) if name_scope else node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None


def getTrainabledNode(graph: Union[None, Graph] = None) -> List:
    graph = DefaultGraph if graph is None else graph
    return [
        node for node in graph.nodes if isinstance(node, Variable) and node.trainable
    ]


def updateNodeValue(
    node_name: str,
    value: np.matrix,
    name_scope: Union[None, str] = None,
    graph: Union[None, Graph] = None,
) -> None:
    node = getNodeByName(node_name, name_scope, graph)
    if node is not None and node.value != value:
        node.value = value


class NameScope(object):
    def __init__(self, name_scope: str) -> None:
        self.name_scope = name_scope

    def __enter__(self) -> None:
        DefaultGraph.name_scope = self.name_scope

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        DefaultGraph.name_scope = None
