# from CSDN
from collections import defaultdict
# 知识点1： dict =defaultdict(factory_function),
# 这个factory_function可以是list、set、str等等，作用是当key不存在时，返回的是工厂函数的默认值，
# 比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0
import random as ra
import numpy as np
import math


# 随机产生数据域内k个初始簇中心点
def generate_k(data, k):
    centers = []
    dimensions = len(data[0])
    min_max = defaultdict(int)  # 创建字典，字典里的key不存在但被查找时，返回的不是keyError,而是一个默认值int 0

    # 找出各个维度的最小/最大的点，即求数据域范围
    for point in data:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i  # 用i替换%d (%i 为整体),输出字符串min_i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    # 随机生成k个点
    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            rand_point.append(ra.uniform(min_val, max_val))  # 在最小和最大范围内取随机点
        centers.append(rand_point)

    return centers


# 平方和误差函数(标准差）
def distance(a, b):
    dimensions = len(a)
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2  # **表指数
        _sum += difference_sq
    return math.sqrt(_sum)


# 将点分配到不同簇中,即分配到距离最近（标准差最小）的簇,center为簇中心的集合
def assign_data_points(data, center):
    assignments = []
    for point in data:
        shortest = 65536  # ?
        shortest_index = 0
        for i in range(len(center)):
            val = (distance(point, center[i]))  # ?
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
        # print(assignments)
    return assignments  # 返回各个点被分配到的簇id的集合序号，与原来的data顺序相对应（非map）


# 求数据的中心
def avg_data_center(data):
    dimensions = len(data[0])
    center = []
    for dimension in range(dimensions):
        _sum = 0
        data_len = len(data)
        for i in range(data_len):  # data的个数
            _sum += data[i][dimension]
        center.append(_sum / len(data))
    return np.array(center)


# 更新簇的均值，即中心,data为点云
def update_data_center(data, target_names):
    new_means = defaultdict(list)  # key不存在时,不存在时返回[]
    center = []
    # 知识点2：zip用法：a=[1,2,3],b=[4,5,6],zp=zip(a,b),print(*zp) 返回：(1, 4) (2, 5) (3, 6)
    for target_names, point in zip(target_names, data):
        new_means[target_names].append(point)

    for data in new_means.values():
        center.append(avg_data_center(data))  # data 为点云

    return center
