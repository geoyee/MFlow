import sys

sys.path.append("E:/dataFiles/github/MFlow")

from mflow_serving import MFlowServer


if __name__ == "__main__":
    root_dir = ""
    model_file_name = ""
    weights_file_name = ""
    host = "localhost:5000"
    # 打开服务
    serving = MFlowServer(host, root_dir, model_file_name, weights_file_name)
    serving.serve()
