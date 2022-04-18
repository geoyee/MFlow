import numpy as np
from .base import Metric
from typing import Any


# 正确率
class Accuracy(Metric):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Accuracy, self).__init__(*parents, **kwargs)

    def init(self) -> None:
        self.correct_num = 0
        self.total_num = 0

    def calcValue(self) -> None:
        # 第一个节点是预测值，第二个节点是标签
        pred = Metric.prob2Label(self.nparents[0].value)
        gt = self.nparents[1].value
        # 正确的数量
        self.correct_num += np.sum(pred == gt)
        # 样本总数
        self.total_num += len(pred)
        self.value = 0
        if self.total_num != 0:
            self.value = float(self.correct_num) / self.total_num


# 准确率
class Precision(Metric):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Precision, self).__init__(*parents, **kwargs)

    def init(self) -> None:
        self.true_pos_nom = 0
        self.pred_pos_num = 0

    def calcValue(self) -> None:
        # 第一个节点是预测值，第二个节点是标签
        pred = Metric.prob2Label(self.nparents[0].value)
        gt = self.nparents[1].value
        # 预测为1的样本数
        self.pred_pos_num += np.sum(pred == 1)
        # 预测为1且正确
        self.true_pos_nom += np.sum(pred == gt and pred == 1)
        self.value = 0
        if self.pred_pos_num != 0:
            self.value = float(self.true_pos_nom) / self.pred_pos_num


# 召回率
class Recall(Metric):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(Recall, self).__init__(*parents, **kwargs)

    def init(self) -> None:
        self.gt_pos_nom = 0
        self.true_pos_nom = 0

    def calcValue(self) -> None:
        # 第一个节点是预测值，第二个节点是标签
        pred = Metric.prob2Label(self.nparents[0].value)
        gt = self.nparents[1].value
        # 标签为1的样本数
        self.gt_pos_nom += np.sum(gt == 1)
        # 预测为1且正确
        self.true_pos_nom += np.sum(pred == gt and pred == 1)
        self.value = 0
        if self.gt_pos_nom != 0:
            self.value = float(self.true_pos_nom) / self.gt_pos_nom


class ROC(Metric):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(ROC, self).__init__(*parents, **kwargs)

    def init(self) -> None:
        self.count = 100
        self.gt_pos_num = 0
        self.gt_neg_num = 0
        self.true_pos_num = 0
        self.false_pos_num = 0
        self.tpr = np.array([0] * self.count)
        self.fpr = np.array([0] * self.count)

    def calcValue(self) -> None:
        # 第一个节点是预测值，第二个节点是标签
        prob = self.nparents[0].value
        gt = self.nparents[0].value
        self.gt_pos_num += np.sum(gt == 1)
        self.gt_neg_num += np.sum(gt == -1)
        # 99个阈值
        thresholds = list(np.arange(0.01, 1.00, 0.01))
        # 分别使用多个阈值产生的预测与标签比较
        for idx, threshold in enumerate(thresholds):
            pred = Metric.prob2Label(prob, threshold)
            self.true_pos_num[idx] += np.sum(pred == gt and pred == 1)
            self.false_pos_num[idx] += np.sum(pred != gt and pred == 1)
        # 分别计算TPR和FPR
        if self.gt_pos_num != 0 and self.gt_neg_num != 0:
            self.tpr = self.true_pos_num / self.gt_pos_num
            self.fpr = self.false_pos_num / self.gt_neg_num

    def valueStr(self) -> str:
        return "{}: [TPR: {}, FPR: {}]}".format(
            self.__class__.__name__, self.tpr, self.fpr
        )


class ROC_AUC(Metric):
    def __init__(self, *parents: Any, **kwargs: Any) -> None:
        super(ROC_AUC, self).__init__(*parents, **kwargs)

    def init(self) -> None:
        self.gt_pos_preds = []
        self.gt_neg_preds = []

    def calcValue(self) -> None:
        # 第一个节点是预测值，第二个节点是标签
        prob = self.nparents[0].value
        gt = self.nparents[0].value
        if gt[0, 0] == 1:
            self.gt_pos_preds.append(prob)
        else:
            self.gt_neg_preds.append(prob)
        total = len(self.gt_pos_preds) * len(self.gt_neg_preds)
        count = 0
        for gt_neg_pred in self.gt_pos_preds:
            for gt_pos_pred in self.gt_pos_preds:
                if gt_pos_pred > gt_neg_pred:
                    count += 1
        self.value = float(count) / total
