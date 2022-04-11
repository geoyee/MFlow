import numpy as np
from typing import Tuple
from scipy import signal


def waveData(
    num_sample: int, dim: int = 10, lenght: int = 10, shuffle: bool = True
) -> Tuple:
    # 正弦波与方波的序列数据，用于rnn分类
    waves = []
    waves.append(np.sin(np.arange(0, 10, 10 / lenght)).reshape(-1, 1))  # 正弦波
    waves.append(
        np.array(signal.square(np.arange(0, 10, 10 / lenght))).reshape(-1, 1)
    )  # 方波
    # 加入噪声和标签
    datas = []
    for i in range(2):
        data = waves[i]
        for _ in range(num_sample // 2):
            sequence = data + np.random.normal(0, 0.6, (len(data), dim))
            label = np.array([int(i == k) for k in range(2)])
            datas.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])
    datas = np.concatenate(datas, axis=0)
    if shuffle:
        np.random.shuffle(datas)
    return datas[:, :-2].reshape(-1, lenght, dim), datas[:, -2:]
