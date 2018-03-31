# coding=utf-8
import numpy as np


# 定义神经网络结构
class Network(object):
    def __init__(self, sizes):
        # [3,2,1] 输入层3个神经元，隐藏层2个，输出层1个 定义总共有多少层，每一层有多少个神经元
        # 网络层数: 一共有多少层
        self.num_layers = len(sizes)
        # 每层神经元的个数
        self.sizes = sizes
        # 初始化每层的偏置 b
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # self.biases = []
        # # 去size从第一项到最后一项 sizes=[3,2,1] 我们只取2,1
        # for y in sizes[1:]:
        #     # 第一次循环时y为2,第二次为1  输入到隐藏，隐藏到输出: 一共两个偏置
        #     # random.randn 使用标准正态分布来初始化一个数组, 初始化一个y乘以1的数组
        #     # 从输入层到隐藏层有两个偏置,隐藏层到输出层有一个偏置
        #     self.biases.append(np.random.randn(y, 1))

        # 初始化每层的权重 w
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # self.weights = []
        # # 使用zip同时遍历两个列表
        # for x, y in zip(sizes[:-1], sizes[1:]):
        #     # sizes = [3,2,1]
        #     # 输入层到隐藏层的连线总共有6条: 第一次我们x取3,y取2
        #     # 隐藏层到输出层的连线有2条, x我们取2,y取1
        #     self.weights.append(np.random.randn(y, x))


# sigmoid激励函数(1/1+e的-z次方)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# 创建网络
net = Network([3, 2, 1])
print(net.num_layers)
print(net.sizes)
print(net.biases)
print(net.weights)
print(net.weights[0])
print(net.weights[1])
