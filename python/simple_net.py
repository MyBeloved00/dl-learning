import numpy as np
from activate_function import softmax
from loss_function import cross_entropy_error
from diff_compute import numerical_gradient


class simple_net:
    def __init__(self, t):
        self.t = t
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

    def numerical_gradient(self, x):
        h = 1e-4
        grad = np.zeros_like(self.W)
        # print(self.W) #test
        # print(grad)
        # print(self.W.flat[2])
        for i in range(self.W.size):
            temp = self.W.flat[i]

            self.W.flat[i] = temp + h
            diffval1 = self.loss(x)

            self.W.flat[i] = temp - h
            diffval2 = self.loss(x)

            grad.flat[i] = (diffval1 - diffval2) / (2.0 * h)
            self.W.flat[i] = temp
        return grad


# test the class
t = np.array([0, 0, 1])
net = simple_net(t)
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(net.loss(x))
dW = net.numerical_gradient(x)
print(dW)
