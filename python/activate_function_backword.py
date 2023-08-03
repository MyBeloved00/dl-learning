import numpy as np


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
