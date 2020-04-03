import numpy as np


class SpeciesIndividual:  # 个体
    fitness = 0.0  # 适应度
    distance = 0.0
    rate = 0.0  # rate = point.fitness/totalFitness

    def __init__(self, genes_len, disMap):
        self.genes_len = genes_len
        self.disMap = disMap
        self.genes = np.zeros(genes_len)
        print("SpeciesIndividual.init()")

    def createByRandomGenes(self):  # 初始化基因（随机）
        print("SpeciesIndividual.createByRandomGenes()")
        for i in range(0, self.genes_len):  # 初始化为0～genes_len
            self.genes[i] = i

        for i in range(0, self.genes_len):  # 随机交换,这边改成贪心算法可以提速
            rand = np.random.choice(self.genes_len)
            temp = self.genes[i]
            self.genes[i] = self.genes[rand]
            self.genes[rand] = temp
        # print("genes[] = ",self.genes)

    def calFitness(self):  # 计算适应度 （修改重点）
        totalDist = 0.0
        for i in range(0, self.genes_len):
            curCity = int(self.genes[i])
            nextCity = int(self.genes[(i + 1) % self.genes_len])
            totalDist = totalDist + disMap[curCity][nextCity]
        self.distance = totalDist
        self.fitness = 1.0 / self.distance
        print("totalDist:", totalDist)




species_num = 200  # 种群数
develop_num = 1000  # 进化次数
pcl = 0.6  # 交叉概率下限  P crossover
pch = 0.95  # 交叉概率上限
pm = 0.4  # 变异概率 mutate

# 初始化城市和城市间距
cityPosition = [[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556],
                [3238, 1229], [4196, 1004], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
                [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370],
                [3780, 2212], [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2367],
                [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
                [2370, 2975]]
city_num = len(cityPosition)
print("city_num:", city_num)

disMap = np.zeros((city_num, city_num))  # 距离地图
for i in range(0, city_num):
    for j in range(0, city_num):
        dist = np.sqrt(
            np.power(cityPosition[i][0] - cityPosition[j][0], 2) + np.power((cityPosition[i][1] - cityPosition[j][1]),
                                                                            2))
        disMap[i][j] = dist
        disMap[j][i] = dist

print("dist between city0 and city1: ", disMap[0][1])

indiv = SpeciesIndividual(city_num, disMap)
indiv.createByRandomGenes()
indiv.calFitness()
