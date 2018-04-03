# encoding: utf-8
__author__ = 'mtianyan'
__date__ = '2018/4/3 0003 17:18'

# coding=utf-8
import random
import numpy as np
import json
import sys


class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)


class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        # 网络层数
        self.num_layers = len(sizes)
        # 网络每层神经元个数
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    # 随机梯度下降
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            test_data=None):
        if test_data: n_test = len(test_data)
        # 训练数据总个数
        n = len(training_data)

        # 开始训练 循环每一个epochs
        for j in range(epochs):
            # 洗牌 打乱训练数据
            random.shuffle(training_data)

            # mini_batch
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]


            # 训练mini_batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            print("Epoch %s training complete" % j)

            cost = self.total_cost(training_data, lmbda)
            print("Cost on training data: {}".format(cost))
            accuracy = self.accuracy(training_data, convert=True)
            print("Accuracy on training data: {} / {}".format(accuracy, n))

            if test_data:
                cost = self.total_cost(test_data, lmbda, convert=True)
                print("Cost on test data: {}".format(cost))
                accuracy = self.accuracy(test_data)
                print("Accuracy on test data: {} / {}".format(accuracy, len(test_data)))

    # 更新mini_batch
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        # 保存每层偏倒
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 训练每一个mini_batch
        for x, y in mini_batch:
            delta_nable_b, delta_nabla_w = self.update(x, y)

            # 保存一次训练网络中每层的偏倒
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nable_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 更新权重和偏置 Wn+1 = wn - eta * nw
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


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

        # 反向更新了
        # 计算最后一层的误差
        delta = (self.cost).delta(zs[-1], activations[-1], y)

        # 最后一层权重和偏置的倒数
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 倒数第二层一直到第一层 权重和偏置的倒数
        for l in range(2, self.num_layers):
            z = zs[-l]

            sp = sigmoid_prime(z)

            # 当前层的误差
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp

            # 当前层偏置和权重的倒数
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost


    def cost_derivative(self, output_activation, y):
        return (output_activation - y)


    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':
    import mnist_loader

    traning_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # net = Network([784, 30, 10])
    # net.SGD(traning_data, 30, 10, 0.5, test_data=test_data)

    net = Network([784, 60, 10])
    net.SGD(traning_data, 30, 10, 0.5, 5.0, test_data=test_data)

