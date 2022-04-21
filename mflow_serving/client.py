import grpc
from ..serving import serving_pb2_grpc


class MFlowServingClient(object):
    def __init__(self, host: str) -> None:
        self.stub = serving_pb2_grpc.MFlowServingStub(grpc.insecure_channel(host))
