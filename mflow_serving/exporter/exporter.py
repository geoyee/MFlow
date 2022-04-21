from ...mflow.core import Graph, DefaultGraph, getNodeByName
from typing import Union, Dict


class Exporter(object):
    def __init__(self, graph: Union[Graph, None] = None) -> None:
        self.graph = graph if graph is not None else DefaultGraph

    def signature(self, input_name: str, output_name: str) -> Dict:
        """返回模型服务接口的签名

        Args:
            input_name (str): 输入的节点名称
            output_name (str): 输出的节点名称

        Returns:
            Dict: 模型服务接口签名
        """
        input_var = getNodeByName(input_name, graph=self.graph)
        assert input_var is not None
        output_var = getNodeByName(output_name, graph=self.graph)
        assert output_var is not None
        input_signature = dict()
        input_signature["name"] = input_name
        output_signature = dict()
        output_signature["name"] = output_name
        return {"inputs": input_signature, "outputs": output_signature}
