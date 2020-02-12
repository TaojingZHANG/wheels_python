import threading
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from testUI import Ui_Dialog


class MainUI(QtWidgets.QMainWindow,Ui_Dialog):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_Dialog.__init__(self)
        self.setupUi(self)


class MainPlot:
    def main(self):
        dataPath = '/home/ztj/PRJ3_map/Data_bumper/result/data.csv'
        headers, data = self.csvReader(dataPath)
        a = [[headers.index('dx'), headers.index('dy')],
             [headers.index('S1C'), headers.index('S2C')],
             [headers.index('M1'), headers.index('M2')]]
        # showFig2D(headers, data, 1, 2, 'r*')
        self.showSubFig2D(headers, data, a, 'r*')  # todo:多线程，避免plot阻塞

        print("finish run MainPlot()")

    # 输入：
    #   csv路径
    # 输出：
    #   data:所有数值数据，array形式,n × m形式,每一列为一个维度
    #   headers:所有数据标题，list形式
    def csvReader(self,dataPath):
        csv_data = pd.read_csv(dataPath, sep=',')
        headers = []
        data = []
        for i in range(len(csv_data.axes[1])):
            header = csv_data.axes[1][i]
            headers.append(header)
            data.append(csv_data[header])
        data = np.array(data)  # list to array
        data = data.transpose()  # 行列转置, 成 n × m形式,每一列为一个维度
        return headers, data


    # 输入：
    #   headers:所有数据标题，list形式
    #   data:所有数值数据，array形式,n × m形式,每一列为一个维度
    #   selDimID1:选择的数据第1个维度编号，selectedDimensionsID，0为第一个维度
    #   selDimID2:选择的数据第2个维度编号，selectedDimensionsID
    #   format:颜色格式,如‘b*’
    def showFig2D(self,headers, data, selDimID1, selDimID2, format):
        matrix_shape = data.shape  # 得到二维矩阵的行列数
        cols = matrix_shape[1]  # dims
        n = cols - 1
        if (selDimID1 > n) | (selDimID2 > n) | (selDimID1 < 0) | (selDimID2 < 0):
            print("\033[0;31;47m showFig2D():Import dims error !\033[0m")  # 输出警告
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
    def showSubFig2D(self,headers, data, selDim, format):
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
                print("\033[0;31;47m showFig2D():Import dims error !\033[0m")  # 输出警告
                exit(1)
            plt1 = plt.subplot(figRows, figCols, figID + 1)
            plt.xlabel(headers[selDimID1])
            plt.ylabel(headers[selDimID2])
            x = data[:, selDimID1]
            y = data[:, selDimID2]
            plt1.plot(x, y, format)

        pos = plt.ginput(2)  # 获得plot上点击的位置
        print(pos[0][0])
        print(pos[1])
        plt.show()


# TODO：添加可加入不同条件的二分类/多分类绘图功能，用不同颜色表示


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    plot = MainPlot()

    try:
        thread1 = threading.Thread(target=window.show(),name="thread1",args=(0))
        thread2 = threading.Thread(target=plot.main(),name="thread2",args=(1))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
    except:
        print("cannot start threads")

    print("finish run, 3q")
    sys.exit(app.exec_())



