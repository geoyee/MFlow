import sys
sys.path.append("E:/dataFiles/github/MFlow")

import numpy as np
from mflow import core, ops


x = core.Variable(size=(3, 1), trainable=False)
y = core.Variable(size=(1, 1), trainable=False)
w = core.Variable(size=(1, 3), trainable=True)
b = core.Variable(size=(1, 1), trainable=True)
x.setValue(np.mat([[182], [72], [0.17]]))
y.setValue(np.mat([[1]]))

model = ops.Step(ops.Add(ops.MatMal(w, x), b))
model.forward()

print(model.value)