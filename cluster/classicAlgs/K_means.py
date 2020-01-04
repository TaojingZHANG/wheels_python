from collections import defaultdict

def generate_k(data,k):
    centers=[]
    dimensions = len(data[0])
    min_max = defaultdict(int)#  字典里的key不存在但被查找时，返回的不是keyError,而是一个默认值int 0

    for point in data:
        for i in range(dimensions):
            val=point[i]
