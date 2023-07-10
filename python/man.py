import numpy as np


class Man:
    def __init__(self, name):
        self.name = name

    def hello(self):
        print('hello ' + self.name)


a = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 3.0]])
b = np.array([1.0, 2.0, 3.0])
c = np.array([1.0, 2.0])
print(a.flatten())
s = Man("jack")
s.hello()
ss = Man("sjj")
ss.hello()