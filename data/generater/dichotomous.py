import numpy as np
from typing import Tuple


def sexData(num_sample: int, shuffle: bool = True) -> Tuple:
    # 生成男性和女性的身高体重及体脂率数据，用于性别分类
    male = {
        "height": np.random.normal(171, 6, num_sample),  # 身高
        "weight": np.random.normal(70, 10, num_sample),  # 体重
        "bfr": np.random.normal(16, 2, num_sample),  # 体脂率
        "label": [1] * num_sample,  # 标签
    }
    female = {
        "height": np.random.normal(158, 5, num_sample),
        "weight": np.random.normal(57, 8, num_sample),
        "bfr": np.random.normal(22, 2, num_sample),
        "label": [-1] * 500,
    }
    data = np.array(
        [
            np.concatenate((male["height"], female["height"])),
            np.concatenate((male["weight"], female["weight"])),
            np.concatenate((male["bfr"], female["bfr"])),
            np.concatenate((male["label"], female["label"])),
        ]
    ).T
    if shuffle:
        np.random.shuffle(data)
    return data[:, :-1], data[:, -1]
