import numpy as np


def mean_squared_error(y, t):  # 均方差误差函数
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):  # 交叉熵误差函数
    # 修改为多维度版本
    if y.ndim == 1:  # 如果为[data]则改为[[data]]
        y = y.reshape(1, y.shape[0])
        t = t.reshape(1, t.shape[0])

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size  # 加delta是因为如果参数为0则会出错
    #return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
    # y[np.arange(batch_size), t]生成一个这样的数组[y[i][t[i]]]
    # 因为t中是实际答案（比如7,2），对应的输出为y[i-1]比如t答案是7则将y[7]的提取出来（即输出），然后ln并求和。