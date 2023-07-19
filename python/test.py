import numpy as np


def relu(x):
    return np.maximum(0, x)


def soft_max(x):
    return np.exp(x) / np.sum(np.exp(x))


def layer_compute(x, w, b=0):
    a = np.dot(x, w)
    return relu(a + b)


def output_compute(x, w, b=0):
    a = np.dot(x, w)
    # return a + b # 恒等函数
    return softmax(a + b)


network = {"x": np.array([5, 2]),
           "b": np.array([0.1, 0.2, 0.3]), "w": np.array([[1, 3, 5], [2, 4, 6]]),
           "b1": np.array([0.1, 0.2, 0.3]), "w1": np.array([[2, 3, 4], [3, 4, 5], [2, 5, 7]]),
           "b2": np.array([0.1, 0.2]), "w2": np.array([[5, 6], [3, 4], [4, 6]])}
layer1 = layer_compute(network["x"], network["w"], network["b"])
layer2 = layer_compute(layer1, network["w1"], network["b1"])
y = output_compute(layer2, network["w2"], network["b2"])  # 输出层的激活函数不太一样，称作恒等函数
print(y)
