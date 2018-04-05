# encoding: utf-8
__author__ = 'mtianyan'
__date__ = '2018/4/5 0005 16:31'
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from numpy import random

# 训练参数
learning_rate = 0.01
training_epochs = 1000
display_step = 50
model_path = "./tmp/model.ckpt"

# 训练数据
train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

# 定义两个变量op占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 初始化模型里所有的w和b
W = tf.Variable(random.random(), name="weight")
b = tf.Variable(random.random(), name="bias")

# 构造线性模型
pred = tf.add(tf.multiply(X, W), b)

# 均方误差
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
# tf.reduce_sum(x) ==> 6
# tf.reduce_sum(x, 0) ==> [2, 2, 2]
# tf.reduce_sum(x, 1) ==> [3, 3]
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

saver = tf.train.Saver()

print("Starting 1st session...")
# 使用session 启用默认图
with tf.Session() as sess:
    sess.run(init)

    # 训练开始
    for epoch in range(200):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # 每个一个epoch打印一下结果
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)


print("Starting 2st session...")
# 使用session 启用默认图
with tf.Session() as sess:
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % save_path)

    # 训练开始
    for epoch in range(training_epochs-200):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # 每个一个epoch打印一下结果
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                  "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # 图画
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.savefig('save_restore_linear_train.png')
    plt.show()

