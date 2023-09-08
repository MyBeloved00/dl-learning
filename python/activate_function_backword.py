import numpy as np
from activate_function import softmax as sm
from loss_function import cross_entropy_error


class RuLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        out = 1 / (1 + np.exp(x))
        self.y = out
        return out

    def backward(self, dout):
        return dout * self.y * (1.0 - self.y)


class softmax:
    def __init__(self):
        self.t = None
        self.y = None
        self.loss = None

    def forward(self, x, t):
        self.t = t
        self.y = sm(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, t, dout=1):
        self.t = t
        batch_size = self.t.shape[0]
        dx = (np.argmax(self.y) - self.t) / batch_size
        return dx
