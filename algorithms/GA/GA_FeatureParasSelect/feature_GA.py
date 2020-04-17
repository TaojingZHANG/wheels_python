import numpy as np
import copy
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from sklearn.model_selection import train_test_split


class SpeciesIndividual:  # 个体
    fitness = 0.0  # 适应度
    rate = 0.0  # rate = self.fitness/totalFitness 选择的概率

    def __init__(self, genes_num, data):
        self.genes_num = genes_num
        self.data = data
        self.genes = np.zeros(genes_num)
        # print("SpeciesIndividual.init()")

    def createByRandomGenes(self):  # 初始化基因（随机）
        # print("SpeciesIndividual.createByRandomGenes()")
        for i in range(0, self.genes_num):  # 随机交换,这边改成贪心算法可以提速
            rand = np.random.choice(20) - 10  # 系数选在-10 ～ 10之间
            self.genes[i] = rand
        # print("genes[] = ",self.genes)

    def calFitness(self):  # 计算适应度 （修改重点）
        newFeature = self.data["newFeature"] * 0
        for i in range(0, self.genes_num):
            newFeature = newFeature + self.data.iloc[:, i] * self.genes[i]
        self.data["newFeature"] = newFeature
        columns = ["ok", "newFeature"]
        data_numeric = self.data[columns]
        correlation = data_numeric.corr()
        self.fitness = correlation.iloc[[0]].values[0][1]
        # print("totalDist:", totalDist)


class GeneticAlgorithm:
    species_num = 50  # 种群个体数
    develop_num = 500  # 进化次数
    pc = 0.1  # 交叉概率  P crossover
    pm = 0.4  # 变异概率 mutate

    def __init__(self, data, gene_num):
        self.data = data
        self.genes_num = gene_num  # 基因个数
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
            species = SpeciesIndividual(self.genes_num, self.data)
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
        for c_i in range(int(self.species_num / 4), self.species_num - 1):
            random_rate = np.random.choice(100) / 100
            if random_rate < self.pc:  # 一定概率下进行交叉，可能不交叉
                sec = np.random.choice(self.species_num)
                species = self.list[c_i]
                species_next = self.list[sec]
                # print("start species: ", species.genes)
                # print("start species_next: ", species_next.genes)

                begin_pos = np.random.choice(self.genes_num)
                end_pos = np.random.choice(self.genes_num)
                # print("begin_pos: ", begin_pos)
                for i in range(begin_pos, end_pos):  # 两个个体交换一段基因
                    # temp = species.genes[i]
                    species.genes[i] = species_next.genes[i]

                    # print("     species: ", species.genes)
                # print("     species_next: ", species_next.genes)

    def mutate(self):  # 变异操作，每个物种都有概率变异
        for i in range(0, self.species_num):
            species = self.list[i]
            for mutate_times in range(0, 3):  # 变异2次
                random_rate = np.random.choice(100) / 100
                # print("start species: ", species.genes)
                if random_rate < self.pm:  # 一定概率变异,随机改变两个基因
                    left = np.random.choice(self.genes_num)
                    right = np.random.choice(self.genes_num)
                    species.genes[left] = np.random.choice(20) - 10 #系数选在-10 ～ 10之间
                    species.genes[right] = np.random.choice(20) - 10
                    # print("end species: ", species.genes)

    def getBest(self):
        fitness_best = self.list[0].fitness
        bestSpecied = self.list[0]
        for i in range(0, self.species_num):
            species = self.list[i]
            if fitness_best < species.fitness:
                fitness_best = species.fitness
                bestSpecied = species

        print("itr:", self.itr, ",  best fitness = ", fitness_best)
        for i in range(0, self.genes_num):
            # print(i, ":", bestSpecied.genes[i], "  ", end="")
            print(bestSpecied.genes[i], "  ", end="")
        print("\n")
        return bestSpecied


# 程序目的：根据已知特征构造新特征，使新特征和结果（ok)之间的相关系数最大，lightGBM用
# TODO:现在程序可能陷入局部最优，很长一段时间fitness无法上升

Train_data = pd.read_csv('/home/ztj/PRJ3_map/wheels_python/algorithms/GA/GA_FeatureParasSelect/data/relo_W_noPredE_less.csv', sep=',')

data_all = pd.concat([Train_data], ignore_index=True, sort=False)  # [Train_data, Train_data2, Train_data3, Train_data4]

data_all_less = data_all
# data_all_more, data_all_less = train_test_split(data_all, test_size=0.2) # 选部分数据，会使结果变化


# 看一下某一项的相关系数
# columns = ["ok", "dFRowLen0"]
# data_numeric = data[columns]
# correlation = data_numeric.corr()
# print("correlation = ", correlation.iloc[[0]].values[0][1])


# 测试构造出来的相关系数
# kp = [6, 2, 6, 9, 3, 6, 3, 6, 8, 8, 3, 5, 1, 1, 7, 7, 2, 3]
# data_all["construct"]=data_all["ok"]*0
# construct=data_all["construct"]
# for i in range(0,18):
#     construct=construct+kp[i]*data_all.iloc[:, i+115]
# data_all["construct"]=construct

# columns = ["ok", "construct"]
# data_numeric = data_all[columns]
# correlation = data_numeric.corr()
#
# f, ax = plt.subplots(figsize=(len(columns), len(columns)))
# sns.heatmap(correlation, square=True, vmax=0.8, annot=True) #显示相关系数
# plt.pause(0.05)
# plt.show()


# 取data_all的77项之后的特征，不取所有特征进行构造
data = data_all_less.iloc[:, 77:data_all_less.columns.size]  # data_all_less.columns.size
data["newFeature"] = data.iloc[:, 1] * 0
data["ok"] = data_all_less.iloc[:, 0]

print("data.size:", data.shape[0])

gene_num = data.columns.size - 1
GA = GeneticAlgorithm(data, gene_num)
GA.run()

print("run finish")
