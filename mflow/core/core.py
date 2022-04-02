from .node import Variable
from .graph import DefaultGraph


def getNodeFromGraph(node_name, name_scope=None, graph=None):
    graph = DefaultGraph if graph is None else graph
    node_name = (name_scope + "/" + node_name) if name_scope else node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None

def getTrainabledFromGraph(graph=None):
    graph = DefaultGraph if graph is None else graph
    return [node for node in graph.nodes if isinstance(node, Variable) and node.trainable]

def updateNodeValueInGraph(node_name, value, name_scope=None, graph=None):
    node = getNodeFromGraph(node_name, name_scope, graph)
    if node is not None and node.value != value:
        node.value = value


class name_scope(object):
    def __init__(self, name_scope):
        self.name_scope = name_scope

    def __enter__(self):
        DefaultGraph.name_scope = name_scope

    def __exit__(self, exc_type, exc_value, exc_tb):
        DefaultGraph.name_scope = None