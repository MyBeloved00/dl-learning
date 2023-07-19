import numpy as np


def mean_squared_error(y, t):  # 均方差误差函数
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):  # 交叉熵误差函数
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) #加delta是因为如果参数为0则会出错
