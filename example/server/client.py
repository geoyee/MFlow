import sys

sys.path.append("E:/dataFiles/github/MFlow")

import numpy as np
from mflow_serving import MFlowServingClient


if __name__ == "__main__":
    # 数据
    test_data = []
    test_label = []
    # 使用推理服务IP和端口，实例化调用客户端
    host = "localhost:5000"
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
