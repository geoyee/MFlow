import grpc
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .proto import serving_pb2_grpc, serving_pb2
from ...mflow.core import getNodeByName
from ...mflow.engine import Saver


class MFlowServingSercive(serving_pb2_grpc.MFlowServingServicer):
    """推理服务
    1. 接受模型文件中定义的接口签名，从计算图中获取输入节点和输出节点
    2. 接受网络请求并解析模型输入
    3. 调用计算图进行计算
    4. 获取输出节点的值并返回给窗口的调用者
    """

    def __init__(self, root_dir, name):
        self.root_dir = root_dir
        self.model_file_name = name + ".json"
        self.weights_file_name = name + ".npz"
        saver = Saver(self.root_dir)
        # 从文件中加载和还原计算图参数，同时获取服务接口签名
        _, service = saver.load(name)
        inputs = service.get("inputs", None)
        outputs = service.get("outputs", None)
        # 根据服务签名中记录的名称，从计算图中查找输入和输出节点
        self.input_node = getNodeByName(inputs["name"])
        self.input_dim = self.input_node.dim
        self.output_node = getNodeByName(outputs["name"])

    def Predict(self, predict_req, context):
        # 从protobuf反序列化到numpy matrix
        inference_rqe = MFlowServingSercive.deserialize(predict_req)
        # 调用计算图，前向传播计算模型的预测结果
        inference_resp = self._inference(inference_rqe)
        # 将结果序列化为protobuf然后通过网络返回
        predict_resp = MFlowServingSercive.serialize(inference_resp)
        return predict_resp

    def _inference(self, inference_req):
        inference_resp_mat_list = []
        for mat in inference_req:
            self.input_node.setValue(mat.T)
            self.output_node.forward()
            inference_resp_mat_list.append(self.output_node.value)
        return inference_resp_mat_list

    @staticmethod
    def deserialize(predict_req):
        infer_req_mat_list = []
        for proto_mat in predict_req.data:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=np.float32)
            mat = np.reshape(mat, dim)
            infer_req_mat_list.append(mat)
        return infer_req_mat_list

    @staticmethod
    def serialize(inference_resp):
        resp = serving_pb2.PredictResp()
        for mat in inference_resp:
            proto_mat = resp.data.add()
            proto_mat.value.extend(np.array(mat).flatten())
            proto_mat.dim.extend(list(mat.shape))
        return resp


# 简易封装
class MFlowServer(object):
    def __init__(
        self, host: str, root_dir: str, name: str, max_workders: int = 10
    ) -> None:
        self.host = host
        self.max_workders = max_workders
        # 实例化gRPC server类
        self.server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workders))
        serving_pb2_grpc.add_MFlowServingServicer_to_server(
            MFlowServingSercive(root_dir, name), self.server
        )
        # 传入监听ip和端口
        self.server.add_insecure_port(self.host)

    def serve(self) -> None:
        # 启动RPC服务
        self.server.start()
        print("MFlow server running on {}.".format(self.host))
        # 永久阻塞等待请求
        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            self.server.stop(0)
