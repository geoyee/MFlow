import numpy as np
import grpc
from .serving import serving_pb2_grpc, serving_pb2
from typing import List, Any


class MFlowServingClient(object):
    def __init__(self, host: str) -> None:
        self.stub = serving_pb2_grpc.MFlowServingStub(grpc.insecure_channel(host))
        print("[GRPC] Connected to MFlow serving: {}.".format(host))

    def Predict(self, mat_data_list: List) -> Any:
        req = serving_pb2.PredictReq()
        for mat in mat_data_list:
            proto_mat = req.data.add()
            proto_mat.value.extend(np.array(mat).flatten())
            proto_mat.dim.extend(list(mat.shape))
        # 调用远程服务的Predict接口
        resp = self.stub.Predict(req)
        return resp
