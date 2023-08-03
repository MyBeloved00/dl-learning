import numpy as np


class affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        y = np.dot(self.x, self.w) + self.b
        return y

    def backward(self, dy):
        self.db = np.sum(dy, axis=0)
        self.dw = np.dot(self.x.T, dy)
        dx = np.dot(dy, self.w.T)

        return dx
