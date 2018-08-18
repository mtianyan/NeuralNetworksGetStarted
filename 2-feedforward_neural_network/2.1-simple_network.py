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


def sigmoid(z):
    """sigmoid激励函数(1/1+e的-z次方)"""
    return 1.0 / (1.0 + np.exp(-z))


if __name__ == '__main__':
    net = Network([3, 2, 1])
    print("网络层数: ", net.num_layers - 1)
    print("网络结构: ", net.sizes)
    print("*" * 20)
    print("输入到隐藏层偏置: ", net.biases[0])
    print("隐藏到输出层偏置: ", net.biases[1])
    print("*" * 20)
    print("输入到隐藏层权重: ", net.weights[0])
    print("隐藏到输出层权重: ", net.weights[1])
