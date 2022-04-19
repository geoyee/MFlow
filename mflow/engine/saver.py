import datetime
import json
import os
import os.path as osp
import numpy as np
from ..core import Node, Variable, Graph, DefaultGraph, getNodeByName
from ..utils import ClassMining
from typing import Dict, Union


class Saver(object):
    def __init__(self, root_dir: str = "") -> None:
        self.root_dir = root_dir
        if not osp.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def save(
        self,
        name: Union[str, None] = None,
        graph: Union[Graph, None] = None,
        meta: Union[Dict, None] = None,
        service: Union[Dict, None] = None,
    ) -> None:
        if graph is None:
            graph = DefaultGraph
        else:
            name = graph.name_scope if graph.name_scope is not None else "output"
        # 元信息，如时间之类的信息
        meta = {} if meta is None else meta
        meta["save_time"] = str(datetime.datetime.now())
        meta["file_name"] = name
        # 服务器端口描述
        service = {} if service is None else service
        self._saveModelAndWeight(name, graph, meta, service)

    def load(self, name: str, graph: Union[Graph, None] = None) -> Dict:
        if graph is None:
            graph = DefaultGraph
        model_json = {}
        graph_json = []
        weights_dict = dict()
        # 读取计算图结构
        model_file_path = os.path.join(self.root_dir, (name + ".json"))
        with open(model_file_path, "r") as model_file:
            model_json = json.load(model_file)
        # 读取值
        weight_file_path = os.path.join(self.root_dir, (name + ".npz"))
        with open(weight_file_path, "rb") as weight_file:
            weights_npz_files = np.load(weight_file)
            for file_name in weights_npz_files.files:
                weight_file[file_name] = weights_npz_files[file_name]
            weights_npz_files.close()
        # 组合
        graph_json = model_json["graph"]
        self._restoreNodes(graph, graph_json, weights_dict)
        print("Load and restore model from {}.".format(osp.join(self.root_dir, name)))
        self.meta = model_json.get("meta", None)
        self.service = model_json.get("service", None)
        return self.meta, self.service

    def _saveModelAndWeight(
        self, name: str, graph: Graph, meta: Dict, service: Dict
    ) -> None:
        model_json = {"meta": meta, "service": service}
        graph_json = []
        weights_dict = dict()
        # 把节点保存为dict/json格式
        for node in graph.nodes:
            if not node.saved:
                continue
            node_json = {
                "node_type": node.__class__.__name__,
                "name": node.name,
                "parents": [p.name for p in node.nparents],
                "childrens": [c.name for c in node.nchildrens],
                "kwargs": node.kwargs,
            }
            # 保存节点dim信息
            if node.value is not None:
                if isinstance(node.value, np.matrix):
                    node_json["dim"] = node.shape
            graph_json.append(node_json)
            # 如果是Var节点还需要保存值
            if isinstance(node, Variable):
                weights_dict[node.name] = node.value
        model_json["graph"] = graph_json
        # 保存计算图json
        model_file_path = osp.join(self.root_dir, (name + ".json"))
        with open(model_file_path, "w") as model_file:
            json.dump(model_json, model_file, indent=4)
            print("Save model into file: {}.".format(model_file.name))
        # 保存值npz
        weights_file_path = osp.join(self.root_dir, (name + ".npz"))
        with open(weights_file_path, "wb") as weights_file:
            np.savez(weights_file, **weights_dict)
            print("Save weights to file: {}.".format(weights_file.name))

    def _restoreNodes(self, graph: Graph, model_json: Dict, weights_dict: Dict) -> None:
        for idx in range(len(model_json.keys())):
            node_json = model_json[idx]
            node_name = node_json["name"]
            weights = None
            if node_name in weights_dict.keys():
                weights = weights_dict[node_name]
            # 判断是否创建了当前节点，存在就更新权值，不存在就创建
            target_node = getNodeByName(node_name, graph=graph)
            if target_node is None:
                print(
                    "Target node {} of type {} not exists, try to create the instance".format(
                        node_json["name"], node_json["node_type"]
                    )
                )
                target_node = Saver.createNode(graph, model_json, node_json)
            target_node.value = weights

    @staticmethod
    def createNode(graph: Graph, model_json: Dict, node_json: Dict) -> None:
        # 递归创建不存在的节点
        node_type = node_json["node_type"]
        node_name = node_json["name"]
        parents_name = node_json["parents"]
        dim = node_json.get("dim", None)
        kwargs = node_json.get("kwargs", None)
        kwargs["graph"] = graph
        parents = []
        for parent_name in parents_name:
            parent_node = getNodeByName(parent_name, graph=graph)
            if parent_node is None:
                parent_node_json = None
                for node in model_json:
                    if node["name"] == parent_name:
                        parent_node_json = node
                assert parent_node_json is not None
                # 父节点不存在就递归调用
                parent_node = Saver.createNode(graph, model_json, parent_node_json)
            parents.append(parent_node)
        # 反射创建节点实例
        if node_type == "Variable":
            assert dim is not None
            dim = tuple(dim)
            return ClassMining.getInstanceBySubclassName(Node, node_type)(
                *parents, dim=dim, name=node_name, **kwargs
            )
        else:
            return ClassMining.getInstanceBySubclassName(Node, node_type)(
                *parents, name=node_name, **kwargs
            )
