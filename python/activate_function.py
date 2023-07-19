import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    y = x > 0
    return y.astype("int")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# x = np.arange(-5, 5, 0.1)
# y1 = step_function(x)
# y2 = sigmoid(x)
# y3 = relu(x)
# plt.plot(x, y1, label="step_function")
# plt.plot(x, y2, linestyle="--", label="sigmoid")
# plt.plot(x, y3, linestyle="solid", label="relu")
# plt.ylim(-0.1, 2)
# plt.legend()
# plt.show()
