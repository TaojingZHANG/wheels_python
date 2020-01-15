import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 准备数据
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N * K, D)) # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8') # class labels
for j in range(K):
    ix = list(range(N*j, N*(j + 1)))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j # 打标签

# 将y转化为one-hot编码
#y = (np.arange(K) == y[:,None]).astype(int)
y = np.eye(K)[y]
# =====================================================


from keras import models
from keras import layers

# 使用Sequential类定义两层模型
model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(2,)))
model.add(layers.Dense(3, activation='softmax'))

# 编译。指定模型的优化器、损失函数、监控指标。
# 对于一个两类分类问题，您将使用二元交叉熵（binary crossentropy）
# 对于一个多类分类问题使用分类交叉熵（categorical crossentropy）
# 对于回归问题使用均方差（meansquared error）
# 对于序列学习问题使用连接主义时间分类（connectionist temporal classification, CTC）
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
from keras import optimizers
from keras import losses
from keras import metrics
#from keras import optimizers, losses, metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
'''

# 训练网络
model.fit(X, y, batch_size=50, epochs=1000)


# =====================================================
# 重新生成数据
X = np.zeros((N * K, D)) # data matrix (each row = single example)
y = np.zeros(N * K, dtype='uint8') # class labels
for j in range(K):
    ix = list(range(N*j, N*(j + 1)))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j # 打标签

# 将y转化为one-hot编码
#y = (np.arange(K) == y[:,None]).astype(int)
y = np.eye(K)[y]
# =====================================================

#检查模型在测试集上的表现是否良好
test_loss, test_acc = model.evaluate(X, y)
print('test_acc:', test_acc)
