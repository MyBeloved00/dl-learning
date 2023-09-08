import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grad):
        for key in params.keys():
            params[key] -= grad[key] * self.lr
