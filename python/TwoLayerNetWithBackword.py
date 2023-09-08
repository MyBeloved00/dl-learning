import numpy as np
from collections import OrderedDict
from affine import affine as af
from activate_function_backword import RuLU as ru
from activate_function_backword import softmax as sm
from gradient_descent import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init=0.01):
        self.params = {}
        self.params["w1"] = np.random.randn(input_size, hidden_size) * weight_init
        self.params["b1"] = np.zeros(hidden_size) * weight_init
        self.params["w2"] = np.random.randn(hidden_size, output_size) * weight_init
        self.params["b2"] = np.zeros(output_size) * weight_init

        self.layers = OrderedDict()  # 生成有序字典，即字典内的元素按照放入顺序排放
        self.layers["affine1"] = af(self.params["w1"], self.params["b1"])
        self.layers["Relu1"] = ru()
        self.layers["affine2"] = af(self.params["w2"], self.params["b2"])

        self.last_layer = sm()

    def predict(self, x):
        for i in self.layers.values():
            x = i.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
            accuracy = np.sum(y == t) / float(x.shape[0])
            return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def backward_gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(t, dout)

        layers = list(self.layers.values())
        layers.reverse()  # 对列表中的元素进行反向排序
        for i in layers:
            dout = i.backward(dout)

        grads = {}
        grads['w1'] = self.layers['affine1'].dw
        grads['b1'] = self.layers['affine1'].db
        grads['w2'] = self.layers['affine2'].dw
        grads['b2'] = self.layers['affine2'].db
        return grads
