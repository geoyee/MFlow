from .base import Trainer
from ..core import Variable
from ..ops import Loss
from ..opts import Optimizer
from typing import List, Union


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
