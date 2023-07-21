import numpy as np
import matplotlib.pyplot as plt


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_diff(f, x):
    h = 1.0e-4
    return (f(x + h) - f(x - h)) / (2.0 * h)


def numerical_gradient(f, x):  # x = w
    h = 1e-4
    x = x.flatten()
    grad = np.zeros_like(x)
    print(x)
    for i in range(x.size):
        temp = x[i]

        x[i] = temp + h
        diffval1 = f(x)

        x[i] = temp - h
        diffval2 = f(x)

        grad[i] = (diffval1 - diffval2) / (2.0 * h)
        x[i] = temp
    return grad.reshape(2, 3)


def slope(f, x, point):  # 生成一条f在x范围内，关于point点的切线
    k = numerical_diff(f, point)
    b = f(point) - k * point
    return k * x + b

# show the image of function_1
# x = np.arange(0, 20, 0.1)
# y = function_1(x)
# k1 = numerical_diff(function_1, 5)
# plt.xlabel("x")
# plt.ylabel("y(x)")
# plt.plot(x, y, label="Function1")
# plt.plot(x, slope(function_1, x, 5), label="slope of Function1 in 5")
# plt.plot(x, slope(function_1, x, 10), label="slope of Function1 in 10")
# plt.legend()
# plt.show()

# test numerical_gradient
# res = numerical_gradient(function_2, np.array([3.0, 4.0]))
# print(res)
# res = numerical_gradient(function_2, np.array([0.0,2.0]))
# print(res)
# res = numerical_gradient(function_2, np.array([3.0, 0.0]))
# print(res)
