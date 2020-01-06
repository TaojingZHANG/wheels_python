# from CSDN
from collections import defaultdict
import random as ra


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


#平方和误差函数
def distance(a,b):
    dimensions = len(a)
    