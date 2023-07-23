import numpy as np
from TwoLayerNet import TwoLayerNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)

iters_num = 20  # 迭代次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
net = TwoLayerNet(784, 50, 10)

train_loss_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = net.numerial_gradient(x_batch, t_batch)

    for k in ("w1", "b1", "w2", "b2"):
        net.params[k] -= grad[k] * learning_rate

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(i)
    print(loss)

print("end")
