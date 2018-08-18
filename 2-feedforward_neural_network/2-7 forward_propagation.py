import random
import numpy as np


class Network(object):
    """神经网络类"""

    def __init__(self, sizes):
        """
        初始化构造方法
        :param sizes: 列表; 如[3,2,1] 定义输入层有3个神经元，隐藏层2个，输出层1个；这定义总共有多少层，每一层有多少个神经元。
        """
        # 网络层数: 一共有多少层
        self.num_layers = len(sizes)
        # 每层神经元的个数
        self.sizes = sizes
        # 初始化每层的偏置 b
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        '''
        上面这行代码的等价写法
        self.biases = []
        for y in sizes[1:]:
            self.biases.append(np.random.randn(y, 1))  # [一个(2,1), 一个(1,1)]

        size[1: ]; sizes=[3,2,1]; 我们只取2,1两个值，第一次循环时y为2,第二次为1, 表示输入到隐藏, 隐藏到输出,一共两种偏置。
        random.randn使用标准正态分布来初始化一个数组,,初始化一个y乘以1的数组,即初始化一个(2,1)的和一个(1,1)的，从输入层到隐藏层有两个偏置,隐藏层到输出层有一个偏置
        self.biases.append(np.random.randn(y, 1))
        '''

        # 初始化每层的权重 w
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        '''
        上面这行代码的等价写法
        self.weights = []
        for x, y in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(y, x)) # 输入层到隐藏层的连线总共有6条(2,3); 隐藏层到输出层的连线有2条(1,2)
        '''

    def update(self, x, y):
        """ 前向传播 过程"""
        # 传入输入的训练数据,
        activation = x

        # 保存每一层的激励值a=sigmoid(z) z=wx+b
        # 第0层(输入层)时输入数据就是它的激励值
        activations = [x]

        # zs用于保存每一层的z=wx+b
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

            # 计算每层经过激活函数后的输出
            activation = sigmoid(z)

            # 保存每一层的a
            activations.append(activation)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
