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

df_2 = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_A_noPredE_less.csv', skiprows=1,
                   header=None,
                   sep=',')

df_3 = pd.read_csv('/home/ztj/PRJ3_map/testRelo/python_analysis/data/Fea10/relo_C_noPredE_less.csv', skiprows=1,
                   header=None,
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
    'learning_rate': 0.08,  # 0.05  # 学习速率3
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    # 'max_depth': 6,
    'bagging_freq': 5,  # 5 # k 意味着每 k 次迭代执行bagging
    'verbose': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
#
print('Start training...')
# 训练 train


bestAuc = 0
print("调参1：提高准确率")
num_leaves_best = 0
max_depth_best = 0
learning_rate_best = 0
for learning_rate in [0.13, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]:#0.05, 0.1,
    for num_leaves in range(40, 90, 3):
        for max_depth in range(5, 8, 1):
            params['learning_rate'] = learning_rate
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, verbose_eval=False,
                            early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

            Auc = gbm.best_score['valid_0']['auc']
            if Auc > bestAuc:
                print("Auc = ", Auc, "learning_rate = ", learning_rate, "num_leaves = ", num_leaves, "max_depth = ",
                      max_depth)
                bestAuc = Auc
                num_leaves_best = num_leaves
                max_depth_best = max_depth
                learning_rate_best = learning_rate

params['num_leaves'] = num_leaves_best
params['max_depth'] = max_depth_best
params['learning_rate'] = learning_rate_best

max_bin_best = 0
min_data_in_leaf_best = 0
bestAuc = 0
print("调参2：降低过拟合")
for max_bin in range(5, 256, 20):
    for min_data_in_leaf in range(1, 102, 10):
        params['max_bin'] = max_bin
        params['min_data_in_leaf'] = min_data_in_leaf
        gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, verbose_eval=False,
                        early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

        Auc = gbm.best_score['valid_0']['auc']
        if Auc > bestAuc:
            print("Auc = ", Auc, "max_bin = ", max_bin, "min_data_in_leaf = ", min_data_in_leaf)
            bestAuc = Auc
            max_bin_best = max_bin
            min_data_in_leaf_best = min_data_in_leaf

params['max_bin'] = max_bin_best
params['min_data_in_leaf'] = min_data_in_leaf_best

feature_fraction_best = 0
bagging_fraction_best = 0
bagging_freq_best = 0
bestAuc = 0
print("调参3：降低过拟合")
for feature_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
    for bagging_fraction in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_freq in range(0, 30, 5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq
            gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, verbose_eval=False,
                            early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

            Auc = gbm.best_score['valid_0']['auc']
            if Auc > bestAuc:
                print("Auc = ", Auc, "feature_fraction = ", feature_fraction, "bagging_fraction = ", bagging_fraction,
                      "bagging_freq = ", bagging_freq)
                bestAuc = Auc
                feature_fraction_best = feature_fraction
                bagging_fraction_best = bagging_fraction
                bagging_freq_best = bagging_freq

params['feature_fraction'] = feature_fraction_best
params['bagging_fraction'] = bagging_fraction_best
params['bagging_freq'] = bagging_freq_best

lambda_l1_best = 0
lambda_l2_best = 0
bestAuc = 0
print("调参4：降低过拟合")
for lambda_l1 in [1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    for lambda_l2 in [1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 1.0]:
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2
        gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, verbose_eval=False,
                        early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

        Auc = gbm.best_score['valid_0']['auc']
        if Auc > bestAuc:
            print("Auc = ", Auc, "lambda_l1 = ", lambda_l1, "lambda_l2 = ", lambda_l2)
            bestAuc = Auc
            lambda_l1_best = lambda_l1
            lambda_l2_best = lambda_l2

params['lambda_l1'] = lambda_l1_best
params['lambda_l2'] = lambda_l2_best

bestAuc = 0
min_split_gain_best = 0
print("调参5：降低过拟合2")
for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    params['min_split_gain'] = min_split_gain
    gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, verbose_eval=False,
                    early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

    Auc = gbm.best_score['valid_0']['auc']
    if Auc > bestAuc:
        print("Auc = ", Auc, "min_split_gain = ", min_split_gain)
        bestAuc = Auc
        min_split_gain_best = min_split_gain

params['min_split_gain'] = min_split_gain_best

# params={'task': 'train', 'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'auc'}, 'num_leaves': 79, 'learning_rate': 0.08, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 15, 'verbose': 1, 'max_depth': 7, 'max_bin': 5, 'min_data_in_leaf': 21, 'lambda_l1': 0.7, 'lambda_l2': 0.7, 'min_split_gain': 0.0}

#      {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'auc'}, 'num_leaves': 55, 'learning_rate': 0.08, 'feature_fraction': 0.8, 'bagging_fraction': 0.7, 'bagging_freq': 20, 'verbose': 1, 'max_depth': 7, 'max_bin': 5, 'min_data_in_leaf': 91, 'lambda_l1': 1.0, 'lambda_l2': 0.7, 'min_split_gain': 0.0}


print("best Params : ")
print(params)
# print(num_leaves_best)
# print(max_depth_best)
# print(max_bin_best)
# print(min_data_in_leaf_best)
# print(feature_fraction_best)
# print(bagging_fraction_best)
# print(bagging_freq_best)
# print(lambda_l1_best)
# print(lambda_l2_best)
# print(min_split_gain_best)


print("best Params : ")
print(params)

gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval,
                early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

# 保存模型到文件
print('Save model...')
gbm.save_model('model_opt.txt')

model_json = gbm.dump_model()
with open('model_opt.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

print('Start predicting...')
# 载入模型文件
gbm = lgb.Booster(model_file='model_opt.txt')

# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 评估模型
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

np.savetxt('/home/ztj/PRJ3_map/testRelo/python_analysis/data/pred.csv', y_pred, delimiter=',')
