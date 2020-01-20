import numpy as np


def nearest_neighbor(src, dst):
    """
       Find the nearest (Euclidean) neighbor in dst for each point in src
       Input:
           src: Nx3 array of points
           dst: Nx3 array of points
       Output:
           distances: Euclidean distances (errors) of the nearest neighbor
           indecies: dst indecies of the nearest neighbor
    """

    a = 0


if __name__ == "__main__":
    A = np.random.randint(0, 101, (20, 3))  # 生成20个随机3d点
