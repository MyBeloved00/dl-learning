import numpy as np
from activate_function import sigmoid, softmax
from diff_compute import numerical_gradient
from loss_function import cross_entropy_error


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params["w1"] = np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["w2"] = np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params["w1"], self.params["w2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)  # z1的维度为hidden_size,求和操作是在loss_function中完成

        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy

    def numerial_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {}
        grads["w1"] = numerical_gradient(loss_w, self.params["w1"])
        grads["b1"] = numerical_gradient(loss_w, self.params["b1"])
        grads["w2"] = numerical_gradient(loss_w, self.params["w2"])
        grads["b2"] = numerical_gradient(loss_w, self.params["b2"])
        return grads

# test
# net = TwoLayerNet(784, 100, 10)
# x = np.random.rand(100, 784)
# t = np.random.rand(100, 10)
# t_index = np.argmax(t, axis=1)
# print(t_index.shape)
# grands_ = net.numerial_gradient(x, t_index)
# y = net.predict(x)
# print(y.shape)
#
# print(grands_["w1"].shape)
# print(grands_["b1"].shape)
# print(grands_["w2"].shape)
# print(grands_["b2"].shape)
