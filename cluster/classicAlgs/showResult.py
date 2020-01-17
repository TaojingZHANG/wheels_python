import numpy as np
import matplotlib.pyplot as plt
import K_means as KM


def show_seed_KM():
    with open('./seeds_dataset.txt') as f:
        read_data = f.read()
        sValues = read_data.split()
        nValues = [float(x) for x in sValues]
        seeds_values = np.array(nValues).reshape(210, 8)
    f.closed

    # 两两特征进行观察
    # figure, ax = plt.subplots(6, 6)
    # for i in range(0, 6):
    #     for j in range(i, 6):
    #         Y = np.array(seeds_values[:, [i, j]])  # 将第i列和第j列放到一个二维array里
    #         for l in range(0, 210):
    #             ax[i][j].scatter(Y[l][0], Y[l][1], s=1, color='green') # scatter:绘制散点图
    #         # print(Y)
    #         # print("_")
    # plt.show()

    # K_Means聚类
    old_assignments = None
    centers = KM.generate_k(seeds_values, 3)
    assignments = KM.assign_data_points(seeds_values, centers)
    cycleNum = 0

    while assignments != old_assignments:
        cycleNum += 1
        new_centers = KM.update_data_center(seeds_values, assignments)
        old_assignments = assignments
        assignments = KM.assign_data_points(seeds_values, new_centers)

    print("cycleNum =", cycleNum)

    # 输出结果
    right1_times1 = 0
    right1_times2 = 0
    right1_times3 = 0
    right2_times1 = 0
    right2_times2 = 0
    right2_times3 = 0
    right3_times1 = 0
    right3_times2 = 0
    right3_times3 = 0
    figure, ax = plt.subplots(6, 6)
    for i in range(0, 6):
        for j in range(i, 6):
            Y = np.array(seeds_values[:, [i, j]])  # 将第i列和第j列放到一个二维array里
            for l in range(0, 210):
                if assignments[l] == 0:
                    ax[i][j].scatter(Y[l][0], Y[l][1], s=1, color='red')  # scatter:绘制散点图
                if assignments[l] == 1:
                    ax[i][j].scatter(Y[l][0], Y[l][1], s=1, color='green')  # scatter:绘制散点图
                if assignments[l] == 2:
                    ax[i][j].scatter(Y[l][0], Y[l][1], s=1, color='blue')  # scatter:绘制散点图
            # print(Y)
            # print("_")

    for l in range(0, 210):
        # print(assignments[l])
        if (seeds_values[l, 7] - 1) == 0:  # TODO:正确率低，可能因为三类没对应上！要小心，所以这边三次都比较了取最大的那个,不一定对
            if assignments[l] == 0:
                right1_times1 += 1
            if assignments[l] == 1:
                right1_times2 += 1
            if assignments[l] == 2:
                right1_times3 += 1

        if (seeds_values[l, 7] - 1) == 2:
            if assignments[l] == 0:
                right2_times1 += 1
            if assignments[l] == 1:
                right2_times2 += 1
            if assignments[l] == 2:
                right2_times3 += 1

        if (seeds_values[l, 7] - 1) == 1:
            if assignments[l] == 0:
                right3_times1 += 1
            if assignments[l] == 1:
                right3_times2 += 1
            if assignments[l] == 2:
                right3_times3 += 1

    right1_times = max(right1_times1, right1_times2, right1_times3)
    right2_times = max(right2_times1, right2_times2, right2_times3)
    right3_times = max(right3_times1, right3_times2, right3_times3)
    right_times = right1_times + right2_times + right3_times

    print("ok times =", right_times, '/', len(seeds_values))
    print("ok1 times =", right1_times)
    print("ok2 times =", right2_times)
    print("ok3 times =", right3_times)

    print("finish run")
    plt.show()


if __name__ == '__main__':
    show_seed_KM()
