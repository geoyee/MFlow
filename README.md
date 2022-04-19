# MFlow

学习[MatrixSlow](https://github.com/zc911/MatrixSlow)的记录。

## 结构

```
MFlow
  ├─ data          # 用于存放一些小的数据集、图像等资源
  ├─ example       # 书上的例子
  └─ mflow         # 对于MatrixSlow的“复现”
       ├─ core     # 计算图、节点等核心代码
       ├─ lays     # 网络的计算层
       ├─ ops      # 基础的算子、激活函数和损失等
       ├─ opts     # 优化器等
       ├─ metrics  # 评估指标等
       ├─ engine   # 任务代码，如训练、评估等
       └─ utils    # 其他等
```
