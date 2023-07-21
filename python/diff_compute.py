import numpy as np
import matplotlib.pyplot as plt


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def numerial_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def slope(f, x, point):#生成一条f在x范围内，关于point点的切线
    k = numerial_diff(f, point)
    b = f(point) - k * point
    return k * x + b


# show the image of function_1
x = np.arange(0, 20, 0.1)
y = function_1(x)
k1 = numerial_diff(function_1, 5)
plt.xlabel("x")
plt.ylabel("y(x)")
plt.plot(x, y, label="Function1")
plt.plot(x, slope(function_1, x, 5), label="slope of Function1 in 5")
plt.plot(x, slope(function_1, x, 10), label="slope of Function1 in 10")
plt.legend()
plt.show()
