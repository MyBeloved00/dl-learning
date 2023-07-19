from dataset.mnist import load_mnist
import pickle
import numpy as np
from activate_function import sigmoid
from activate_function import softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_train, t_train


def ini_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


def predict_batch(network, x_batch):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    for i in x_batch:
        a1 = np.dot(i, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        np.append(y, softmax(a3))
    return y


x, t = get_data()  # 因为one-hot参数为False，所以t为0-9的数字。
network = ini_network()
accuracy = 0
batch_size = 100
for i in range(0, len(x), batch_size):  # 修改为批处理版本
    x_batch = x[i:i + batch_size]
    t_batch = t[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    # 对于二维数组来说，如果axis是0，则是arr[0][i],arr[1][i]......中比较，
    # 如果axis是1,则为arr[i][0],arr[i][1]......中比较
    accuracy = np.sum(p == t_batch)
print(accuracy / len(x))
