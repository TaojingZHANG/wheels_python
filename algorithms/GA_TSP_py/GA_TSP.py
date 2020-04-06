import numpy as np
import copy


class SpeciesIndividual:  # 个体
    fitness = 0.0  # 适应度
    distance = 0.0
    rate = 0.0  # rate = self.fitness/totalFitness 选择的概率

    def __init__(self, genes_len, disMap):
        self.genes_len = genes_len
        self.disMap = disMap
        self.genes = np.zeros(genes_len)
        # print("SpeciesIndividual.init()")

    def createByRandomGenes(self):  # 初始化基因（随机）
        # print("SpeciesIndividual.createByRandomGenes()")
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
        # print("totalDist:", totalDist)


class GeneticAlgorithm:
    species_num = 100  # 种群个体数
    develop_num = 500  # 进化次数
    pcl = 0.9  # 交叉概率下限  P crossover
    pch = 1.0  # 交叉概率上限 ，随机数在0.6~1之间再进行交叉
    pm = 0.4  # 变异概率 mutate

    def __init__(self, cityPosition, disMap):
        self.cityPosition = cityPosition
        self.disMap = disMap
        self.genes_num = len(cityPosition)  # 基因个数即城市个数
        self.list = []  # 种群列表
        self.list_new = []  # 选出的新种群列表
        self.itr = 0

    def run(self):
        self.createBeginningSpeicies()

        for i in range(0, self.develop_num):
            self.itr = i
            self.select()
            self.crossover()
            self.mutate()
            self.getBest()

    def createBeginningSpeicies(self):  # 创建初始种群
        randomNum = self.species_num  # 100%随机生成
        for i in range(0, randomNum):
            species = SpeciesIndividual(self.genes_num, self.disMap)
            species.createByRandomGenes()
            self.list.append(species)

    def calRate(self):  # 计算每个物种被选中的概率，用于轮盘赌算法
        totalFitness = 0.0
        for i in range(0, self.species_num):
            species = self.list[i]
            species.calFitness()
            totalFitness = totalFitness + species.fitness
        # print("totalFitness: ", totalFitness)
        for i in range(0, self.species_num):
            species = self.list[i]
            species.rate = species.fitness / totalFitness
            # print("species.rate: ", species.rate)

    def select(self):  # 最优秀的物种复制占25% + 轮盘赌策略选择剩下75%
        self.calRate()
        fitness_max = 0.0
        best_species = self.list[0]  # 找到最优的物种个体
        self.list_new.clear()  # 注意不要用.clear() 会把list也清了，除非用深拷贝？
        for i in range(0, self.species_num):
            species = self.list[i]
            if species.fitness > fitness_max:
                fitness_max = species.fitness
                best_species = species

        for i in range(0, int(self.species_num / 4)):  # 最优秀的物种复制占25%，优秀的总在前面
            species_new = copy.deepcopy(best_species)  # ！注意！：深拷贝
            self.list_new.append(species_new)

        # 轮盘赌算法:能保证按概率取到规定的n个物种个体。
        # 思路：生成一个随机数，它在概率带的哪个位置，就选择那个个体。
        # 如:共10个个体，则累计概率（总的概率带）为1，当生成随机数为0.3，若前3个个体累计概率0.2，
        #    前4个个体累计概率0.4，则取第4个个体
        len_list_now = len(self.list_new)
        for i in range(0, self.species_num - len_list_now):  # 剩下75%用轮盘赌确定
            random_rate = np.random.choice(10000) / 10000  # 生成的随机概率,最多10000个物种，否则没法选
            rate_all = 0.0  # 个体的累计概率
            get_species = False
            for idx in range(0, self.species_num):
                species = self.list[idx]
                species_new = copy.deepcopy(species)  # ！注意！：深拷贝
                if random_rate < rate_all:  # 在区间内了
                    self.list_new.append(species_new)
                    get_species = True
                    break
                else:
                    rate_all = rate_all + species.rate

            if get_species == False:  # 添加最后一个
                species = self.list[self.species_num - 1]
                species_new = copy.deepcopy(species)  # ！注意！：深拷贝
                self.list_new.append(species_new)
        self.list.clear()
        self.list = copy.deepcopy(self.list_new)
        # print("list_new.size =  ", len(self.list_new))

    def crossover(self):  # 交叉操作（一定概率发生）,对随机某个位置，两个相邻的个体随机同一段位置进行交叉，TODO：可以针对不同问题优化
        for c_i in range(0, self.species_num - 1):
            random_rate = np.random.choice(100) / 100
            if (random_rate > self.pcl) & (random_rate < self.pch):  # 一定概率下进行交叉，可能不交叉
                species = self.list[c_i]
                species_next = self.list[c_i + 1]
                # print("start species: ", species.genes)
                # print("start species_next: ", species_next.genes)

                begin_pos = np.random.choice(self.genes_num)
                # print("begin_pos: ", begin_pos)
                for i in range(begin_pos, self.genes_num):  # 两个个体交换一段基因
                    fir = 0
                    sec = 0
                    # 找出species.genes中与species_next.genes[i]相等的位置fir
                    while species.genes[fir] != species_next.genes[i]:
                        fir = fir + 1
                    # 找出species_next.genes中与species.genes[i]相等的位置sec
                    while species_next.genes[sec] != species.genes[i]:
                        sec = sec + 1
                    # 两个基因互换
                    tmp = species.genes[i]
                    species.genes[i] = species_next.genes[i]
                    species_next.genes[i] = tmp
                    # 消去互换后重复的那个基因
                    species.genes[fir] = species_next.genes[i]
                    species_next.genes[sec] = species.genes[i]
                # print("     species: ", species.genes)
                # print("     species_next: ", species_next.genes)

    def mutate(self):  # 变异操作，每个物种都有概率变异
        for i in range(0, self.species_num):
            species = self.list[i]
            random_rate = np.random.choice(100) / 100
            # print("start species: ", species.genes)
            if random_rate < self.pm:  # 一定概率变异
                # 寻找逆转的左右端点
                left = np.random.choice(self.genes_num)
                right = np.random.choice(self.genes_num)
                if left > right:
                    tmp = left
                    left = right
                    right = tmp

                # 逆转left-right间下标元素
                while left < right:
                    tmp = species.genes[left]
                    species.genes[left] = species.genes[right]
                    species.genes[right] = tmp
                    left = left + 1
                    right = right - 1
                # print("end species: ", species.genes)

    def getBest(self):
        distance_best = self.list[0].distance
        bestSpecied = self.list[0]
        for i in range(0, self.species_num):
            species = self.list[i]
            if distance_best > species.distance:
                distance_best = species.distance
                bestSpecied = species

        print("itr:", self.itr, ",  best Dist = ", distance_best)
        return bestSpecied


cityPosition = [[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556],
                [3238, 1229], [4196, 1004], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
                [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370],
                [3780, 2212], [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2367],
                [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
                [2370, 2975]]  # 初始化城市和城市间距
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

GA = GeneticAlgorithm(cityPosition, disMap)
GA.run()

print("run finish")
