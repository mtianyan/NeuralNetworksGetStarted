# encoding: utf-8
__author__ = 'mtianyan'
__date__ = '2018/3/19 0019 23:54'

import numpy as np

a = np.array([2, 3, 4])
print(a)
# 元素数据类型
print(a.dtype)
# 数组的维度(3,) 一行三列
print(a.shape)
# 数组的维数 一维
print(a.ndim)
# 数组的元素个数
print(a.size)
print("*********************************")

b = np.array([[1, 2], [3, 4]])
print(b)
# 元素数据类型
print(b.dtype)
# 数组的维度(2,2) 两行两列
print(b.shape)
# 数组的维数 一维
print(b.ndim)
# 数组的元素个数
print(b.size)
print("*********************************")

c = np.array([[1, 2], [3, 4]], dtype=float)
print(c)
print("*********************************")

# np.zeros创建零矩阵
d = np.zeros((3, 4))
print(d)
print("*********************************")

# np.ones创建全1矩阵,每个元素初始化为1.0
e = np.ones((3, 4))
print(e)
print("*********************************")

# 首先创建一个两行三列的数组
b = np.ones((2, 3))
print(b)
# reshape成三行两列的数组
print(b.reshape(3, 2))
print("*********************************")

# 如何组合两个数组

# 1-数乘
a = np.ones((3, 4))
# a中的每一项都乘以2，然后赋值给b
b = a * 2
print(a)
print(b)
print("*********************************")

# 2-水平合并:
# 注意传入参数为元组,否则传入a,b不报错也没有结果
print(np.hstack((a, b)))
print("*********************************")

# 3-垂直合并
print(np.vstack((a, b)))




