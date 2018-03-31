# coding=utf-8
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
    # GD函数有两个变量: 训练数据集, 需要训练的轮数
    def GD(self, training_data, epochs):
        # 开始训练 循环每一个epochs: 定义epochs数值为几,就循环几次
        for j in range(epochs):
            # 洗牌 打乱训练数据 shuffle
            random.shuffle(training_data)
            # 让每一次训练的时候,训练数据的顺序不同

            # 训练每一个数据, 使用x和y来取训练数据data和它对应的y
            for x, y in training_data:
                # 使用update方法进行前向传播
                self.update(x, y)

            # 每个epoch 完成,打印我们已经训练到了第几个epoch
            print("Epoch {0} complete".format(j))

    # 前向传播
    def update(self, x, y):
        # 传入输入的训练数据,
        activation = x

        # 保存每一层的激励值a=sigmoid(z) z=wx+b
        # 第一层时输入数据就是它的激励值
        activations = [x]

        # 保存每一层的z=wx+b
        zs = []
        # 前向传播
        # 使用for循环遍历每一层的偏置与权重:同时取第一层的偏置和权重
        for b, w in zip(self.biases, self.weights):
            # 计算每层的z
            # dot是点乘方法: 把两个数组进行点乘,对于二维数组相当于矩阵乘法。
            # 一维数组相当于向量的内积
            z = np.dot(w, activation) + b

            # 保存每层的z
            zs.append(z)

            # 计算每层的a
            activation = sigmoid(z)

            # 保存每一层的a
            activations.append(activation)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

