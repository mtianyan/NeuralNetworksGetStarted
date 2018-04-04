# encoding: utf-8
__author__ = 'mtianyan'
__date__ = '2018/4/4 0004 17:47'
import tensorflow as tf


# # 定义常量op
# a = tf.constant(2)
# b = tf.constant(3)
#
# # 使用seesion 启动默认图
# with tf.Session() as sess:
#     print("a=2, b=3")
#     print("常量相加: %i" % sess.run(a + b))
#     print("常量相乘: %i" % sess.run(a * b))


# # 定义连个变量op占位符
# a = tf.placeholder(tf.int16)
# b = tf.placeholder(tf.int16)
#
# # 定义2个op操作 加法 乘法
# add = tf.add(a, b)
# mul = tf.multiply(a, b)
#
# with tf.Session() as sess:
#     print("加法：%i" % sess.run(add, feed_dict={a: 2, b: 3}))
#     print("乘法：%i" % sess.run(mul, feed_dict={a: 2, b: 3}))


# 1x2 矩阵常量op
matrix1 = tf.constant([[3., 3.]])

# 2x1 矩阵常量op
matrix2 = tf.constant([[2.], [2.]])

# 矩阵乘op
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(type(result))
    print(result)

