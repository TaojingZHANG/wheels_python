# coding: utf-8
# pylint: disable = invalid-name, C0111
import time
import json
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# iris = load_iris()
# data=iris.data
# target = iris.target
# X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.2)


# 加载训练集
print('Load data...')

df_1 = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_DZ_noPredE_less.csv', skiprows=1,
                   header=None,
                   sep=',')

df_2 = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_C_noPredE_less.csv', skiprows=1, header=None,
                   sep=',')

df_3 = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_A_noPredE_less.csv', skiprows=1, header=None,
                   sep=',')

data_train_all = pd.concat([df_1, df_2, df_3], sort=False)


print("df_1 : ", len(df_1))
print("df_2 : ", len(df_2))
print("df_3 : ", len(df_3))
print("data_all : ", len(data_train_all))

# 加载验证集
df_test_all = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_W_noPredE_less.csv', skiprows=1,
                          header=None,
                          sep=',')  # skiprows=1, header=None,
print("df_test_all : ", len(df_test_all))

# 去除训练集中重复数据
print("Before delete duplicate size: ", len(data_train_all))
df_all_delDup = data_train_all.drop_duplicates()
print("After delete duplicate size: ", len(df_all_delDup))

# 设置训练样本
df_train = df_all_delDup
df_test = df_test_all

y_train = df_train[0].values  # 输出ok/error
y_test = df_test[0].values  # 输出ok/error
X_train = df_train.drop(0, axis=1).values  # 训练集特征
X_test = df_test.drop(0, axis=1).values  # 测试集特征

# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 64,  # 31 # 叶子节点数   # 由于lightGBM是leaves_wise生长，官方说法是要小于2^max_depth
    'learning_rate': 0.2,  # 0.05  # 学习速率3
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'max_depth': 7,
    'bagging_freq': 5,  # 5 # k 意味着每 k 次迭代执行bagging
    'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

print('Start training...')
# 训练 train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

print("best Auc = ", gbm.best_score['valid_0']['auc'])

print('Save model...')
# 保存模型到文件
gbm.save_model('model.txt')

model_json = gbm.dump_model()
with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

print('Start predicting...')
# 载入模型文件
gbm = lgb.Booster(model_file='model.txt')
# 预测数据集
start = time.process_time()

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

end = time.process_time()
print(end - start, "s")

# dotfile = lgb.create_tree_digraph(gbm, tree_index=1) #用dotfile画出某一棵树，需要graphviz
# print(dotfile)

# 画出某一棵树
import matplotlib.pyplot as plt

fig2 = plt.figure(figsize=(20, 20))
ax = fig2.subplots()
lgb.plot_tree(gbm, tree_index=0, ax=ax)
plt.show()

# 评估模型
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

np.savetxt('/home/ztj/PRJ3_map/testRelo/python_analysis/data/pred.csv', y_pred, delimiter=',')
