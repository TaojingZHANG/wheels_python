import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 输入：
#   csv路径
# 输出：
#   data:所有数值数据，array形式,n × m形式,每一列为一个维度
#   headers:所有数据标题，list形式
def csvReader(dataPath):
    csv_data = pd.read_csv(dataPath, sep=',')
    headers = []
    data = []
    for i in range(len(csv_data.axes[1])):
        header = csv_data.axes[1][i]
        headers.append(header)
        data.append(csv_data[header])
    data = np.array(data)  # list to array
    data = data.transpose() # 行列转置, 成 n × m形式,每一列为一个维度
    return headers, data


# 输入：
#   headers:所有数据标题，list形式
#   data:所有数值数据，array形式,n × m形式,每一列为一个维度
#   selDimID1:选择的数据第1个维度编号，selectedDimensionsID，0为第一个维度
#   selDimID2:选择的数据第2个维度编号，selectedDimensionsID
#   format:颜色格式,如‘b*’
def showFig2D(headers, data, selDimID1, selDimID2, format):
    matrix_shape = data.shape  # 得到二维矩阵的行列数
    cols = matrix_shape[1]  # dims
    n = cols - 1
    if (selDimID1 > n) | (selDimID2 > n) | (selDimID1 < 0) | (selDimID2 < 0):
        print("\033[0;31;47m showFig2D():Import dims error !\033[0m")
        exit(1)

    fig = plt.figure()
    plt1 = plt.subplot(111)
    plt.xlabel(headers[selDimID1])
    plt.ylabel(headers[selDimID2])
    x = data[:, selDimID1]
    y = data[:, selDimID2]
    plt1.plot(x, y, format)
    plt.show()


# 输入：
#   headers:所有数据标题，list形式
#   data:所有数值数据，array形式,n × m形式,每一列为一个维度
#   selDim:选择的数据维度编号,list形式,如绘制维度1和维度12,维度2和维度3：[[1,12],[2,3]]
#   format:颜色格式,如‘b*’
def showSubFig2D(headers, data, selDim, format):
    matrix_shape = data.shape  # 得到二维矩阵的行列数
    cols = matrix_shape[1]  # dims
    n = cols - 1

    figNums = len(selDim)
    figRows = int(np.sqrt(figNums))
    figCols = np.ceil(figNums / figRows)

    fig = plt.figure()
    for figID in range(figNums):
        selDimID1 = selDim[figID][0]
        selDimID2 = selDim[figID][1]

        if (selDimID1 > n) | (selDimID2 > n) | (selDimID1 < 0) | (selDimID2 < 0):
            print("\033[0;31;47m showFig2D():Import dims error !\033[0m")
            exit(1)
        plt1 = plt.subplot(figRows, figCols, figID + 1)
        plt.xlabel(headers[selDimID1])
        plt.ylabel(headers[selDimID2])
        x = data[:, selDimID1]
        y = data[:, selDimID2]
        plt1.plot(x, y, format) # TODO：添加二分类/多分类的绘图功能
    plt.show()


if __name__ == '__main__':
    dataPath = '/home/ztj/PRJ3_map/Data_bumper/result/data7_7.csv'
    headers, data = csvReader(dataPath)

    a = [[1, 2],[3,4],[5,6],[7,8]]

    # showFig2D(headers, data, 1, 2, 'r*')  # todo:多线程，避免plot阻塞
    showSubFig2D(headers, data, a, 'r*')

    print("finish run , 3q")
