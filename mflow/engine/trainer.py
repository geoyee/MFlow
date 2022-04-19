import numpy as np
from ..core import Variable
from ..ops import Loss
from ..opts import Optimizer
from typing import List, Dict, Union


# 训练基类
class Trainer(object):
    def __init__(
        self,
        input_x: List,
        input_y: Variable,
        loss_op: Loss,
        optimizer: Optimizer,
        epochs: int,
        batch_size: int = 8,
        evalable: bool = True,
        metrics_ops: Union[None, List] = None,
        *args,
        **kwargs
    ) -> None:
        self.inputs = input_x
        self.y = input_y
        self.loss = loss_op
        self.opt = optimizer
        self.epochs = epochs
        self._ep = 0
        self.batch_size = batch_size
        self.evalable = evalable
        self.metrics_ops = metrics_ops

    def trainEval(
        self,
        train_x: Dict,
        train_y: Union[List, np.ndarray],
        val_x: Union[None, Dict] = None,
        val_y: Union[None, List, np.ndarray] = None,
    ) -> None:
        self._initVars()
        self.mainLoop(train_x, train_y, val_x, val_y)

    def mainLoop(
        self,
        train_x: Dict,
        train_y: Union[List, np.ndarray],
        val_x: Union[None, Dict] = None,
        val_y: Union[None, List, np.ndarray] = None,
    ) -> None:
        for self._ep in range(self.epochs):
            self.train(train_x, train_y)
            if self.evalable and val_x is not None and val_y is not None:
                self.eval(val_x, val_y)

    def train(self, train_x: Dict, train_y: Union[List, np.ndarray]) -> None:
        for i in range(len(list(train_x.values())[0])):
            self.step(self._getInputValues(train_x, i), train_y[i])
            if (i + 1) % self.batch_size == 0:
                self._optimizerUpdate()

    # 需要完成metrics节点
    def eval(self, val_x: Dict, val_y: Union[List, np.ndarray]) -> None:
        for metrics_op in self.metrics_ops:
            metrics_op.resetValue()
        for i in range(len(list(val_x.values())[0])):
            self.step(self._getInputValues(val_x, i), val_y[i], True)
            for metrics_op in self.metrics_ops:
                metrics_op.forward()
        # 打印指标
        meterics_str = "[Epoch {0}] eval metrics: ".format(self._ep + 1)
        for meterics_op in self.metrics_ops:
            meterics_str += meterics_op.valueStr() + " "
        print(meterics_str)

    def step(self, data_x, data_y, is_eval: bool = False) -> None:
        for i in range(len(self.inputs)):
            # 根据输入节点的名称进行数据的查找
            input_value = data_x.get(self.inputs[i].name)
            self.inputs[i].setValue(np.mat(input_value).T)
        self.y.setValue(np.mat(data_y).T)
        if not is_eval:
            self.opt.step()

    def _getInputValues(self, x: Dict, index: int) -> Dict:
        input_dict = dict()
        for node_name in x.keys():
            input_dict[node_name] = x[node_name][index]
            return input_dict

    # 由于分布式的话不太相同，因此留给否面实现
    def _initVars(self) -> None:
        raise NotImplementedError()

    def _optimizerUpdate(self) -> None:
        raise NotImplementedError()


# 简单的训练器
class SimpleTrainer(Trainer):
    def __init__(
        self,
        input_x: List,
        input_y: Variable,
        loss_op: Loss,
        optimizer: Optimizer,
        epochs: int,
        batch_size: int = 8,
        evalable: bool = True,
        metrics_ops: Union[None, List] = None,
        *args,
        **kwargs
    ) -> None:
        super(SimpleTrainer, self).__init__(
            input_x,
            input_y,
            loss_op,
            optimizer,
            epochs,
            batch_size,
            evalable,
            metrics_ops,
            *args,
            **kwargs
        )

    def _initVars(self) -> None:
        # 不进行任何初始化
        pass

    def _optimizerUpdate(self) -> None:
        self.opt.update()
