# MFlow

学习[MatrixSlow](https://github.com/zc911/MatrixSlow)的记录。

## 结构

```
MFlow
  ├─ data           # 用于存放一些小的数据集、图像等资源
  ├─ example        # 书上的例子
  ├─ mflow_serving  # 服务端
  └─ mflow          # 对于MatrixSlow的“复现”
       ├─ core      # 计算图、节点等核心代码
       ├─ lays      # 网络的计算层
       ├─ ops       # 基础的算子、激活函数和损失等
       ├─ opts      # 优化器等
       ├─ metrics   # 评估指标等
       ├─ models    # 部分模型等
       ├─ engine    # 任务代码，如训练、评估等
       └─ utils     # 其他等
```

## TODO

- [x] 把例子中的模型放到models中。
- [x] 完善`train`的例子，检查是否存在问题。
- [ ] 卷积的“valid”模式存在问题。
- [ ] 完善`server`的例子，检查是否存在问题。
- [ ] 搞清楚`ClassMining`的作用。

完成后向后续学习。
