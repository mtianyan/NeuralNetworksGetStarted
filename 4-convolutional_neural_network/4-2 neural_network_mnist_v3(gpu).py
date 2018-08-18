__author__ = 'mtianyan'
__date__ = '2018/4/3 0003 18:50'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import sigmoid

# 是否使用GPU
GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify " + \
          "network3.py\nto set the GPU flag to False.")
    try:
        theano.config.device = 'gpu'
    except:
        pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify " + \
          "network3.py to set\nthe GPU flag to True.")


class Network(object):
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        # 把每一层的参数放到一个list里
        self.params = [param for layer in self.layers for param in layer.params]
        # 初始化x, y
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        # 初始化第一层
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        # 初始化后面每层
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        # 最后一层的输出
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        num_training_batches = size(training_data) / mini_batch_size
        num_validation_batches = size(validation_data) / mini_batch_size
        num_test_batches = size(test_data) / mini_batch_size

        # l2正则化
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + \
               0.5 * lmbda * l2_norm_squared / num_training_batches
        # 计算梯度
        grads = T.grad(cost, self.params)
        updates = [(param, param - eta * grad)
                   for param, grad in zip(self.params, grads)]

        # 更新方程
        i = T.lscalar()
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                    training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                    test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        # 开始训练
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        # 前一层神经元个数
        self.n_in = n_in
        # 后一层神经元个数
        self.n_out = n_out
        # 激励函数
        self.activation_fn = activation_fn
        # dropout
        self.p_dropout = p_dropout
        # 初始化权重
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        # 初始化偏置
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # reshape输入
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        # 输出
        self.output = self.activation_fn(
            (1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        # 取最大值
        self.y_out = T.argmax(self.output, axis=1)
        # dropout输入dropout
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        # dropout输出
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))


def size(data):
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)
