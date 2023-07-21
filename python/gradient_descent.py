from diff_compute import numerical_gradient
from diff_compute import function_2
import numpy as np


def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


# test
res = gradient_descent(function_2, np.array([-3.0, 4.0]))
print(res)