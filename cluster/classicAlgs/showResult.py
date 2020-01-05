import numpy as np
import matplotlib.pyplot as plt


def show_seed():
    with open('.\seeds_dataset.txt') as f:
        read_data = f.read()
        sValues = read_data.split()
        nValues = [float(x) for x in sValues]
        seeds_values = np.array(nValues).reshape(210, 8)
    f.closed
    figure, ax = plt.subplots(6, 6)
    for i in range(0, 6):
        for j in range(0, 6):
            Y = np.array(seeds_values[:, [i, j]])  # 将第i列和第j列放到一个二维array里
            for l in range(0, 210):
                ax[i][j].scatter(Y[l][0], Y[l][1], s=1, color='mediumseagreen')
            # print(Y)
            # print("_")
    plt.show()


if __name__ == '__main__':
    show_seed()
