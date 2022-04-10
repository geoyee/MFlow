import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from ..core import Graph


def drawXYDatas(datas: List, labs: List) -> None:
    pxs = datas[:, 0]
    pys = datas[:, 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(pxs, pys, c=labs.flatten())
    plt.show()


def drawGraph(graph: Graph) -> None:
    G = nx.Graph()
    already = []
    labels = {}
    for node in graph.nodes:
        G.add_node(node)
        labels[node] = node.__class__.__name__ + ("({:s})".format(str(node.size)) if hasattr(node, "dim") else "") \
            + ("\n[{:.3f}]".format(np.linalg.norm(node.jacobi))
                if node.jacobi is not None else "")
        for c in node.nchildrens:
            if {node, c} not in already:
                G.add_edge(node, c)
                already.append({node, c})
        for p in node.nparents:
            if {node, p} not in already:
                G.add_edge(node, p)
                already.append({node, c})
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.clear()
    ax.axis("on")
    ax.grid(True)
    pos = nx.spring_layout(G, seed=42)
    # 有雅克比的变量节点
    cm = plt.cm.Reds
    nodelist = [n for n in graph.nodes if n.__class__.__name__ ==
                "Variable" and n.jacobi is not None]
    colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, 
                           cmap=cm, edgecolors="#666666",
                           node_size=2000, alpha=1.0, ax=ax)
    # 无雅克比的变量节点
    nodelist = [n for n in graph.nodes if n.__class__.__name__ ==
                "Variable" and n.jacobi is None]
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", 
                           cmap=cm, edgecolors="#666666",
                           node_size=2000, alpha=1.0, ax=ax)
    # 有雅克比的计算节点
    nodelist = [n for n in graph.nodes if n.__class__.__name__ !=
                "Variable" and n.jacobi is not None]
    colorlist = [np.linalg.norm(n.jacobi) for n in nodelist]
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colorlist, 
                           cmap=cm, edgecolors="#666666",
                           node_size=2000, alpha=1.0, ax=ax)
    # 无雅克比的中间
    nodelist = [n for n in graph.nodes if n.__class__.__name__ !=
                "Variable" and n.jacobi is None]
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="#999999", 
                           cmap=cm, edgecolors="#666666",
                           node_size=2000, alpha=1.0, ax=ax)
    # 边
    nx.draw_networkx_edges(G, pos, width=2, edge_color="#014b66", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_weight="bold", 
                            font_color="#6c6c6c", font_size=8,
                            font_family='arial', ax=ax)
    # 显示图像
    plt.show()