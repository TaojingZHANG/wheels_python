import string
import time
import json
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter

# MP分布占总MP的比例

# 加载训练集
print('Load data...')

Train_data1 = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_A_noPredE.csv', sep=',')
Train_data2 = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_C_noPredE.csv', sep=',')
Train_data3 = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_DZ_noPredE.csv', sep=',')
Test_data = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_W_noPredE.csv', sep=',')

# Test_data = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_W_add.csv', sep=',')

# 特征构造
# Train_data['train'] = 1
# Test_data['train'] = 0
# data = pd.concat([Train_data, Test_data], ignore_index=True)

data = pd.concat([Train_data1, Train_data2, Test_data], ignore_index=True, sort=False)

# # 画Hist分析
# figure_i = 0
# for col in data.columns:
#     figure_i = figure_i + 1
#     print("column = ", col)
#     plt.figure(figure_i)
#     data[col].plot.hist()
#     plt.pause(0.05)
#     input()
#     plt.close(figure_i)

# columns = ["FMPNum", "FMPLow", "FlineNum", "FVerticalNum", "FCeilingNum", "FRowNum", "FColNum", "FMaxRowLen",
#            "FMaxColLen", "FdRowColLen", "FMaxRowDist", "FMaxColDist", "FMPPos0", "FMPPos1", "FMPPos2", "FMPPosNumXY0",
#            "FMPPosNumXY1", "FMPPosNumXY2", "FMPPosNumXY3", "FRowDist0", "FRowDist1", "FRowDist2", "FRowDist3",
#            "FRowDist4", "FColDist0", "FColDist1", "FColDist2", "FColDist3", "FColDist4", "FRowLen0", "FRowLen1",
#            "FRowLen2", "FRowLen3", "FRowLen4", "FColLen0", "FColLen1", "FColLen2", "FColLen3", "FColLen4",
#            "closedKFMaxDist", "cloesdKFNum", "LMPNum", "LMPLow", "LlineNum", "LVerticalNum", "LCeilingNum", "LRowNum",
#            "LColNum", "LMaxRowLen", "LMaxColLen", "LdRowColLen", "LMaxRowDist", "LMaxColDist", "LMPPos0", "LMPPos1",
#            "LMPPos2", "LMPPosNumXY0", "LMPPosNumXY1", "LMPPosNumXY2", "LMPPosNumXY3", "LRowDist0", "LRowDist1",
#            "LRowDist2", "LRowDist3", "LRowDist4", "LColDist0", "LColDist1", "LColDist2", "LColDist3", "LColDist4",
#            "LRowLen0", "LRowLen1", "LRowLen2", "LRowLen3", "LRowLen4", "LColLen0", "LColLen1", "LColLen2", "LColLen3",
#            "LColLen4", "dFMPNum", "dFMPLow", "dFlineNum", "dFVerticalNum", "dFCeilingNum", "dFRowNum", "dFColNum",
#            "dFMaxRowLen", "dFMaxColLen", "dFMaxRowDist", "dFMaxColDist", "dFMPPos0", "dFMPPos1", "dFMPPos2",
#            "dFMPPosNumXY0", "dFMPPosNumXY1", "dFMPPosNumXY2", "dFMPPosNumXY3", "dFRowDist0", "dFRowDist1", "dFRowDist2",
#            "dFRowDist3", "dFRowDist4", "dFColDist0", "dFColDist1", "dFColDist2", "dFColDist3", "dFColDist4",
#            "dFRowLen0", "dFRowLen1", "dFRowLen2", "dFRowLen3", "dFRowLen4", "dFColLen0", "dFColLen1", "dFColLen2",
#            "dFColLen3", "dFColLen4", "rFMPNum", "rFMPLow", "rFlineNum", "rFVerticalNum", "rFCeilingNum", "rFRowNum",
#            "rFColNum", "rFMaxRowLen", "rFMaxColLen", "rFMaxRowDist", "rFMaxColDist", "rFMPPos0", "rFMPPos1", "rFMPPos2",
#            "rFMPPosNumXY0", "rFMPPosNumXY1", "rFMPPosNumXY2", "rFMPPosNumXY3"]

# 较好的特征 FMaxRowLen

# columns = ["ok","FMPNum", "FMPLow", "FlineNum", "FVerticalNum", "FCeilingNum", "FRowNum", "FColNum", "FMaxRowLen"]

dropColumn = []
i = 0
# for col in data.columns:
#     cor = np.corrcoef(data["ok"], data[col])
#     if (cor[0, 1] > -0.05) & (cor[0, 1] < 0.05):
#         print("drop ", col, " : ", cor[0, 1])
#         data.drop([col], axis=1)
#         dropColumn.append(col)
#         i = i + 1

# if (cor[0, 1] > 0.15) | (cor[0, 1] < -0.2):
#     print(col, " = ", cor[0, 1])

data_all = []
data_all.append(Train_data1)
data_all.append(Train_data2)
data_all.append(Train_data3)
data_all.append(Test_data)
data_path = []
data_path.append("/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_A_noPredE_less.csv")
data_path.append("/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_C_noPredE_less.csv")
data_path.append("/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_DZ_noPredE_less.csv")
data_path.append("/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_W_noPredE_less.csv")

for i in range(0, 4):
    data = data_all[i]

    # data["rFMaxRowLen"]=data["dFMaxRowLen"]/data["FMaxRowLen"]
    # data["rFMaxColLen"] = data["dFMaxColLen"]/data["FMaxColLen"]
    # data["rFMaxRowDist"] = data["dFMaxRowDist"]/data["FMaxRowDist"]
    # data["rFMaxColDist"] = data["dFMaxColDist"]/data["FMaxColDist"]
    # data["rFRowNum"] = data["dFRowNum"]/data["FRowNum"]
    # data["rFCeilingNum"] = data["dFCeilingNum"]/data["FCeilingNum"]
    # data["rFVerticalNum"] = data["dFVerticalNum"]/data["FVerticalNum"]

    # 较差    的特征
    # columns = ["ok","dFRowLen0", "dFRowLen1", "dFRowLen2", "dFRowLen3","dFColLen0", "dFColLen1", "dFColLen3"]
    # columns = ["ok", "dFRowDist3", "dFColDist0","dFColDist1",  "dFColDist3","dFMPPosNumXY2","dFMPNum", "rFMPNum"]
    #  "rFMPNum", "dFColLen3","dFRowLen0", "dFRowLen1", "dFRowLen2", "dFRowLen3","dFColLen0", "dFColLen1",
    #  "dFRowDist3", "dFColDist0", "dFColDist1", "dFColDist3"
    #  "dFMPPosNumXY2","dFMPNum"

    # 特征构造1 -0.14
    data["dRowDist_add"] = data["dFRowDist1"] + data["dFRowDist2"] + data["dFRowDist3"] + data[
        "dFRowDist4"]
    #
    # columns = ["ok", "dRowDist_add", "dFRowDist0", "dFRowDist1", "dFRowDist2", "dFRowDist3", "dFRowDist4"]

    # 特征构造2 -0.24  无用
    # data["dRowLen_add"] = data["dFRowLen0"] + data["dFRowLen1"] + data["dFRowLen2"] + data["dFRowLen3"] + data[
    #     "dFRowLen4"]
    #
    # columns = ["ok", "dRowLen_add", "dFRowLen0", "dFRowLen1", "dFRowLen2", "dFRowLen3", "dFRowLen4"]

    # 特征构造3 -0.17

    data["dColLen_add"] = data["dFColLen0"] + data["dFColLen1"] + data["dFColLen2"] + data["dFColLen3"] + data[
        "dFColLen4"]
    #
    # columns = ["ok", "dColLen_add", "dFColLen0", "dFColLen1", "dFColLen2", "dFColLen3", "dFColLen4"]

    # 特征构造4 -0.15
    data["dColDist_add"] = data["dFColDist1"] + data["dFColDist2"] + data["dFColDist3"] + data[
        "dFColDist4"]
    # columns = ["ok", "dColDist_add", "dFColDist0", "dFColDist1", "dFColDist2", "dFColDist3", "dFColDist4"]

    # 特征构造5 -0.38
    data["dmaxLen_add"] = data["rFMaxRowLen"] + data["rFMaxColLen"]
    # columns = ["ok", "dmaxLen_add","rFMaxRowLen", "rFMaxColLen", "rFMaxRowDist", "rFMaxColDist"]

    # #特征构造6 -0.28
    data["dmaxDist_add"] = data["rFMaxRowDist"] + data["rFMaxColDist"]
    # columns = ["ok", "dmaxDist_add","rFMaxRowLen", "rFMaxColLen", "rFMaxRowDist", "rFMaxColDist"]
    #
    # 特征构造7 -0.41
    # data["dmaxLenDist_add"] = data["rFMaxRowLen"] + data["rFMaxColLen"] + data["rFMaxRowDist"] + data["rFMaxColDist"]
    # columns = ["ok", "dmaxLenDist_add","rFMaxRowLen", "rFMaxColLen", "rFMaxRowDist", "rFMaxColDist"]

    # 特征构造8 -0.48 和上面那个相关
    data["r_all"] = data["rFMaxRowLen"] + data["rFMaxColLen"] + data["rFMaxRowDist"] + data["rFMaxColDist"] + data[
        "rFRowNum"] + data["rFCeilingNum"] + data["rFVerticalNum"]

    # columns = ["ok", "r_all", "rFMaxRowLen", "rFMaxColLen", "rFMaxRowDist", "rFMaxColDist"]

    # 特征构造9 -0.43～-0.53
    data["d_all"] = (data["dFVerticalNum"] + data["dFCeilingNum"]  + data["dFRowNum"] + data["dFMaxRowLen"] / 5 \
                     + data["dFMaxColLen"] / 5 + data["dFMaxRowDist"] / 5)

    # columns = ["ok", "d_all", "dFVerticalNum", "dFCeilingNum", "dFRowNum", "dFMaxRowLen", "dFMaxColLen", "dFMaxRowDist"]
    # columns = ["ok", "dmaxLen_add", "dmaxLenDist_add", "r_all", "d_all"]

    # # #特征构造10 0.24
    # data["F_all"] = data["FCeilingNum"] / (max(data["FCeilingNum"])) + data["FRowNum"] / (max(data["FRowNum"])) \
    #                 + data["FMaxRowLen"] / (max(data["FMaxRowLen"])) + data["FMaxRowDist"] / (max(data["FMaxRowDist"])) \
    #                 + data["FRowLen4"] / (max(data["FRowLen4"]))

    #
    # columns = ["ok", "F_all", "FCeilingNum", "FRowNum", "FMaxRowLen", "FMaxRowDist", "FRowLen4"]

    # for drop in dropColumn:
    #     data = data.drop([drop], axis=1)
    #     print("drop : ", drop)

    data = data.drop(["closedKFMaxDist"], axis=1)
    data = data.drop(["cloesdKFNum"], axis=1)



    # data = data.drop(["dFMPNum"], axis=1)
    # data = data.drop(["dFMPLow"], axis=1)
    # data = data.drop(["dFlineNum"], axis=1)




    # # 特征构造10 0.24

    data["FrMPPosNum0"] = round(100 * data["FMPPosNumXY0"] / (data["FMPNum"]))
    data["FrMPPosNum1"] = round(100 * data["FMPPosNumXY1"] / (data["FMPNum"]))
    data["FrMPPosNum2"] = round(100 * data["FMPPosNumXY2"] / (data["FMPNum"]))
    data["FrMPPosNum3"] = round(100 * data["FMPPosNumXY3"] / (data["FMPNum"]))

    data["LrMPPosNum0"] = round(100 * data["LMPPosNumXY0"] / (data["LMPNum"]))
    data["LrMPPosNum1"] = round(100 * data["LMPPosNumXY1"] / (data["LMPNum"]))
    data["LrMPPosNum2"] = round(100 * data["LMPPosNumXY2"] / (data["LMPNum"]))
    data["LrMPPosNum3"] = round(100 * data["LMPPosNumXY3"] / (data["LMPNum"]))
    #
    data["drMPPosNum0"] = abs(data["FrMPPosNum0"] - data["LrMPPosNum0"])
    data["drMPPosNum1"] = abs(data["FrMPPosNum1"] - data["LrMPPosNum1"])
    data["drMPPosNum2"] = abs(data["FrMPPosNum2"] - data["LrMPPosNum2"])
    data["drMPPosNum3"] = abs(data["FrMPPosNum3"] - data["LrMPPosNum3"])

    # columns = ["ok", "drMPPosNum0", "drMPPosNum1", "drMPPosNum2", "drMPPosNum3","dFMPPosNumXY0", "dFMPPosNumXY1", "dFMPPosNumXY2", "dFMPPosNumXY3"]
    #
    #
    # # 特征间相关系数 绘图
    # data_numeric = data[columns]
    # correlation = data_numeric.corr()
    #
    # f, ax = plt.subplots(figsize=(len(columns), len(columns)))
    # sns.heatmap(correlation, square=True, vmax=0.8, annot=True)
    # plt.pause(0.05)
    # plt.show()

    # 保存特征
    print("start save csv : ", i)
    df = pd.DataFrame(data)
    df.to_csv(data_path[i], header=True, index=False)
    print("finish save csv : ", i)
