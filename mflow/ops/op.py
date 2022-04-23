import numpy as np
from typing import Any
from .base import Operator
from ..core import Node


""" 注意
在每个算子类的计算重载后
请将数据的类型使用`astype("float32")`进行转换
可以避免numpy的一些错误
"""


# 将filler矩阵填充在to_be_filled的对角线上
def fillDiag(to_be_filled: np.matrix, filler: np.matrix) -> np.matrix:
    assert (
        to_be_filled.shape[0] / filler.shape[0]
        == to_be_filled.shape[1] / filler.shape[1]
    )
    n = int(to_be_filled.shape[0] / filler.shape[0])
    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r : (i + 1) * r, i * c : (i + 1) * c] = filler
    return to_be_filled


class Add(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Add, self).__init__(*parents, **kargs)

    def calcValue(self) -> None:
        self.value = np.mat(np.zeros(self.nparents[0].shape))
        for parent in self.nparents:
            self.value += parent.value

    def calcJacobi(self, parent: Any) -> np.matrix:
        return np.mat(np.eye(self.dim)).astype("float32")


# 矩阵乘法
class MatMul(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(MatMul, self).__init__(*parents, **kargs)

    def calcValue(self) -> None:
        assert (
            len(self.nparents) == 2
            and self.nparents[0].shape[1] == self.nparents[1].shape[0]
        )
        self.value = (self.nparents[0].value * self.nparents[1].value).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        zeros = np.mat(np.zeros((self.dim, parent.dim)))
        if parent is self.nparents[0]:
            return fillDiag(zeros, self.nparents[1].value.T)
        else:
            jacobi = fillDiag(zeros, self.nparents[0].value)
            row_sort = np.arange(self.dim).reshape(self.shape[::-1]).T.ravel()
            col_sort = np.arange(parent.dim).reshape(parent.shape[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort].astype("float32")


# 对应位置的元素相乘
class Multiply(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Multiply, self).__init__(*parents, **kargs)

    def calcValue(self) -> None:
        self.value = np.multiply(self.nparents[0].value, self.nparents[1].value).astype(
            "float32"
        )

    def calcJacobi(self, parent: Any) -> np.matrix:
        if parent is self.nparents[0]:
            return np.diag(self.nparents[1].value.A1).astype("float32")
        else:
            return np.diag(self.nparents[0].value.A1).astype("float32")


class ScalarMultiply(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(ScalarMultiply, self).__init__(*parents, **kargs)

    def calcValue(self) -> None:
        assert self.nparents[0].shape == (1, 1)  # 第一个父节点是标量
        self.value = np.multiply(self.nparents[0].value, self.nparents[1].value)

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent in self.nparents
        if parent is self.nparents[0]:
            return self.nparents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.nparents[1].dim)) * self.nparents[0].value[0, 0]


class Reshape(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Reshape, self).__init__(*parents, **kargs)
        self.new_shape = tuple(kargs.get("shape"))
        assert len(self.new_shape) == 2

    def calcValue(self) -> None:
        self.value = self.nparents[0].value.reshape(self.new_shape)

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent is self.nparents[0]
        return np.mat(np.eye(self.dim)).astype("float32")


class Concat(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Concat, self).__init__(*parents, **kargs)
        self.axis = kargs.get("axis", 1)

    def calcValue(self) -> None:
        self.value = np.concatenate(
            [p.value.flatten() for p in self.nparents], axis=self.axis
        ).T

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent in self.nparents
        dims = [p.dim for p in self.nparents]
        index = self.nparents.index(parent)
        dim = int(parent.dim)
        assert dim == dims[index]
        jacobi = np.mat(np.zeros((self.dim, dim)))
        start_row = int(np.sum(dims[:index]))
        jacobi[start_row : start_row + dim, 0:dim] = np.eye(dim)
        return jacobi.astype("float32")


# 焊接
class Welding(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Welding, self).__init__(*parents, **kargs)

    def calcValue(self) -> None:
        assert len(self.nparents) == 1 and self.nparents[0] is not None
        self.value = self.nparents[0].value

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent is self.nparents[0]
        return np.mat(np.eye(self.dim)).astype("float32")

    def weld(self, node: Node) -> None:
        # 与之前的父节点断开
        if len(self.nparents) == 1 and self.nparents[0] is not None:
            self.nparents[0].nchildrens.remove(self)
        self.nparents.clear()
        # 与传入节点焊接
        self.nparents.append(node)
        node.nchildrens.append(self)


# 卷积
class Convolve(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(Convolve, self).__init__(*parents, **kargs)
        assert len(self.nparents) == 2  # 图像与卷积核
        self.padded = None
        self.padding = kargs.get("padding", "valid")
        if self.padding != "valid":
            self.padding = "same"

    def calcValue(self) -> None:
        data = self.nparents[0].value
        kernel = self.nparents[1].value
        self.w, self.h = data.shape
        self.kw, self.kh = kernel.shape
        self.hkw, self.hkh = int(self.kw / 2), int(self.kh / 2)
        if self.padding == "same":
            self.pw, self.ph = tuple(
                np.add(data.shape, np.multiply((self.hkw, self.hkh), 2))
            )
            # 数据补充
            self.padded = np.mat(np.zeros((self.pw, self.ph)))
            self.padded[
                self.hkw : self.hkw + self.w, self.hkh : self.hkh + self.h
            ] = data
            # 结果
            self.value = np.mat(np.zeros((self.w, self.h))).astype("float32")
            # 范围
            self.i_range = np.arange(self.hkw, self.w + self.hkw)
            self.j_range = np.arange(self.hkh, self.h + self.hkh)
        else:  # self.padding == "valid"
            self.pw, self.ph = data.shape
            self.padded = data
            self.value = np.mat(
                np.zeros((self.w - 2 * self.hkw, self.h - 2 * self.hkh))
            ).astype("float32")
            self.i_range = np.arange(self.hkw, self.w - self.hkw)
            self.j_range = np.arange(self.hkh, self.h - self.hkh)
        # 开始卷积
        for i in self.i_range:
            for j in self.j_range:
                self.value[i - self.hkw, j - self.hkh] = np.sum(
                    np.multiply(
                        self.padded[
                            i - self.hkw : i - self.hkw + self.kw,
                            j - self.hkh : j - self.hkh + self.kh,
                        ],
                        kernel,
                    )
                )

    def calcJacobi(self, parent: Any) -> np.matrix:
        kernel = self.nparents[1].value
        jacobi = []
        if parent is self.nparents[0]:  # 图像
            for i in self.i_range:
                for j in self.j_range:
                    mask = np.mat(np.zeros((self.pw, self.ph)))
                    mask[
                        i - self.hkw : i - self.hkw + self.kw,
                        j - self.hkh : j - self.hkh + self.kh,
                    ] = kernel
                    if self.padding == "same":
                        jacobi.append(
                            mask[
                                self.hkw : self.w + self.hkw,
                                self.hkh : self.h + self.hkh,
                            ].A1
                        )
                    else:  # self.padding == "valid"
                        jacobi.append(
                            mask[
                                self.hkw : self.w - self.hkw,
                                self.hkh : self.h - self.hkh,
                            ].A1
                        )
        elif parent is self.nparents[1]:  # 卷积核
            for i in self.i_range:
                for j in self.j_range:
                    jacobi.append(
                        self.padded[
                            i - self.hkw : i - self.hkw + self.kw,
                            j - self.hkh : j - self.hkh + self.kh,
                        ].A1
                    )
        else:
            raise Exception("It's not {0}'s father node.".format(self.name))
        return np.mat(jacobi).astype("float32")


class MaxPooling(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(MaxPooling, self).__init__(*parents, **kargs)
        self.stride = tuple(kargs.get("stride"))
        assert len(self.stride) == 2
        self.size = tuple(kargs.get("size"))
        assert len(self.size) == 2
        self.flag = None

    def calcValue(self) -> None:
        data = self.nparents[0].value
        w, h = data.shape
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size
        hkw, hkh = int(kw / 2), int(kh / 2)
        result = []
        flag = []
        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口的最大值
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                row.append(np.max(window))
                # 记录最大值在原特征图的位置
                pos = np.argmax(window)
                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)
            result.append(row)
        self.flag = np.mat(flag).astype("float32")
        self.value = np.mat(result).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent is self.nparents[0] and self.jacobi is not None
        return self.flag


class AvePooling(Operator):
    def __init__(self, *parents: Any, **kargs: Any) -> None:
        super(AvePooling, self).__init__(*parents, **kargs)
        self.stride = tuple(kargs.get("stride"))
        assert len(self.stride) == 2
        self.size = tuple(kargs.get("size"))
        assert len(self.size) == 2
        self.flag = None

    def calcValue(self) -> None:
        data = self.nparents[0].value
        w, h = data.shape
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size
        hkw, hkh = int(kw / 2), int(kh / 2)
        result = []
        flag = []
        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口的平均值
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                ave = np.mean(window)
                row.append(ave)
                # 平均值还原
                ww, wh = window.shape
                w_dim = ww * wh
                tmp = ave * w_dim * np.ones(dim)
                flag.append(tmp)
            result.append(row)
        self.flag = np.mat(flag).astype("float32")
        self.value = np.mat(result).astype("float32")

    def calcJacobi(self, parent: Any) -> np.matrix:
        assert parent is self.nparents[0] and self.jacobi is not None
        return self.flag
