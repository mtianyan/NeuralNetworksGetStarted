# encoding: utf-8
__author__ = 'mtianyan'
__date__ = '2018/3/29 0029 22:20'
import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        # 网络层数
        self.num_layers = len(sizes)
        # 网络每层神经元个数
        self.sizes = sizes
        # 初始化每层的偏置
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 初始化每层的权重
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # 梯度下降
    def GD(self, training_data, epochs, eta):
        # 开始训练 循环每一个epochs
        for j in range(epochs):
            # 洗牌 打乱训练数据
            random.shuffle(training_data)

            # 反向: 保存每层偏导
            # 反向: 取到每一层的偏置值，取到它的形状，以这个形状创建零矩阵
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            # 训练每一个数据
            for x, y in training_data:
                delta_nable_b, delta_nabla_w = self.update(x, y)

                # 保存一次训练网络中每层的偏倒
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nable_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            # 更新权重和偏置 Wn+1 = wn - eta * nw
            self.weights = [w - (eta) * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta) * nb
                           for b, nb in zip(self.biases, nabla_b)]

            print ("Epoch {0} complete".format(j))

    # 前向传播
    def update(self, x, y):
        # 保存每层偏倒
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x

        # 保存每一层的激励值a=sigmoid(z)
        activations = [x]

        # 保存每一层的z=wx+b
        zs = []
        # 前向传播
        for b, w in zip(self.biases, self.weights):
            # 计算每层的z
            z = np.dot(w, activation) + b

            # 保存每层的z
            zs.append(z)

            # 计算每层的a
            activation = sigmoid(z)

            # 保存每一层的a
            activations.append(activation)

        # 反向更新了: 从倒数第一层开始
        # 计算最后一层的误差
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        # 最后一层权重和偏置的倒数
        # 偏loos/偏b = delta
        # 偏loss/偏w = 倒数第二层y 乘以 delta
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 倒数第二层一直到第一层 权重和偏置的倒数
        for l in range(2, self.num_layers):
            # zs[-2]倒数第二层
            z = zs[-l]

            # 计算倒数第二层的偏导
            sp = sigmoid_prime(z)

            # 当前层的误差: delta_h公式 上一层的w乘以上一层的误差，点乘于本层计算出来的z
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            # 当前层偏置和权重的倒数
            nabla_b[-l] = delta
            # 当前层误差乘以前一层y  -l-1前一层
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        # 返回当前层的偏置和权重的导数
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activation, y):
        return (output_activation - y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


