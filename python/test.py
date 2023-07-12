import numpy as np


def relu(x):
    return np.maximum(0, x)


def layer_compute(x, w):
    a = np.dot(x, w)
    return relu(a + b)


x = np.array([1, 5, 2])  # 第一个输入为bias的系数
b = np.array([0.1, 0.2, 0.3])
w = np.array([b, [1, 3, 5], [2, 4, 6]])
w2 = np.array([[2, 3, 4], [3, 4, 5], [2, 5, 7]])
w3 = np.array([[5, 6], [3, 4], [4, 6]])
layer1 = layer_compute(x, w)
print(layer1)
layer2 = layer_compute(layer1, w2)
print(layer2)
print(w3.shape)
y = layer_compute(layer2, w3)
print(y)
