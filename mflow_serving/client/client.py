import numpy as np
import grpc
from ..serving import serving_pb2_grpc, serving_pb2
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
        # 通过壮调用远程服务的Predict接口
        resp = self.stub.Predict(req)
        return resp


if __name__ == "__main__":
    # 填入数据
    test_data = []
    test_label = []
    # 使用推理服务IP和端口，实例化调用客户端
    host = "127.0.0.1:5000"
    client = MFlowServingClient(host)
    # 遍历测试数据，调用服务进行预测和打印
    for idx in range(len(test_data)):
        img = test_data[idx]
        label = test_label[idx]
        resp = client.Predict([img])
        resp_mat_list = []
        for proto_mat in resp.data:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=np.float32)
            mat = np.reshape(mat, dim)
            resp_mat_list.append(mat)
    pred = np.argmax(resp_mat_list[0])
    print("model predict {} and ground truth: {}.".format(np.argmax(pred.value), label))
