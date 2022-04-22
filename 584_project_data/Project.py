import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.style.use('seaborn')
import tensorflow as tf
# import seaborn as sns
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as ms

# 数据初始化
orders = pd.read_csv("orders_all_join_tab.csv", keep_default_na=False, na_filter=True)
devices = pd.read_csv('device_repeat_tab.csv', keep_default_na=False, na_filter=True)

# 计算注册时间和下单时间差值
orders['ordermins'] = pd.to_datetime(orders['order_date'])
orders['joinmins'] = pd.to_datetime(orders['join_date'])
orders['RegisteredTime'] = (orders['ordermins'] - orders['joinmins']).dt.seconds

devices_s_df = devices[['device_id', 'repeat']]
ret_df = pd.merge(orders, devices_s_df, how='inner', on='device_id')

orders = ret_df.drop(columns=['Unnamed: 0', 'device_id'], axis=1)

# 归一化处理
orders['RegisteredTime'] = StandardScaler().fit_transform(orders[['RegisteredTime']])
orders['refund_rate'] = StandardScaler().fit_transform(orders[['refund_rate']])
orders['browsing_time'] = StandardScaler().fit_transform(orders[['browsing_time']])
orders['count'] = StandardScaler().fit_transform(orders[['count']])
# orders['repeat'] = StandardScaler().fit_transform(orders[['repeat']])

certificate_len = []
for x in orders.index:
    if len(orders.loc[x, 'certificate']) > 9:
        certificate_len.append(0)
    else:
        certificate_len.append(1)

orders['certificate_len'] = certificate_len
orders = orders.drop(columns=['certificate'], axis=1)

orders['certificate_len'] = StandardScaler().fit_transform(orders[['certificate_len']])

target = orders['flag'].values

feature = orders[['browsing_time', 'refund_rate', 'RegisteredTime', 'count', 'certificate_len','flag']]

# 准备训练和测试集
mask = (feature['flag'] == 0)

X_train, X_test = ms.train_test_split(feature[mask], test_size=0.25, random_state=42)
X_train = X_train.drop(['flag'], axis=1).values
X_test = X_test.drop(['flag'], axis=1).values

# 提取所有正样本，作为测试集的一部分
X_fraud = feature[~mask].drop(['flag'], axis=1).values

# 构建Autoencoder网络模型
# 隐藏层节点数分别为16，8，8，16
# epoch为5，batch size为32
input_dim = X_train.shape[1]
encoding_dim = 16
num_epoch = 5
batch_size = 32

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['mae'])

# 模型保存为model.h5，并开始训练模型
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
history = autoencoder.fit(X_train, X_train,
                          epochs=num_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1,
                          callbacks=[checkpointer]).history

# 画出损失函数曲线
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(history['loss'], c='dodgerblue', lw=3)
plt.plot(history['val_loss'], c='coral', lw=3)
plt.title('model loss')
plt.ylabel('mse');
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.subplot(122)
plt.plot(history['mae'], c='dodgerblue', lw=3)
plt.plot(history['val_mae'], c='coral', lw=3)
plt.title('model mae')
plt.ylabel('mae');
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

# 读取模型
autoencoder = load_model('model.h5')

# 利用autoencoder重建测试集
pred_test = autoencoder.predict(X_test)
# 重建欺诈样本
pred_fraud = autoencoder.predict(X_fraud)

# 计算重构MSE和MAE误差
mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
mse_fraud = np.mean(np.power(X_fraud - pred_fraud, 2), axis=1)
mae_test = np.mean(np.abs(X_test - pred_test), axis=1)
mae_fraud = np.mean(np.abs(X_fraud - pred_fraud), axis=1)
mse_df = pd.DataFrame()
mse_df['Class'] = [0] * len(mse_test) + [1] * len(mse_fraud)
mse_df['MSE'] = np.hstack([mse_test, mse_fraud])
mse_df['MAE'] = np.hstack([mae_test, mae_fraud])
mse_df = mse_df.sample(frac=1).reset_index(drop=True)

# 分别画出测试集中正样本和负样本的还原误差MAE和MSE
markers = ['o', '^']
markers = ['o', '^']
colors = ['dodgerblue', 'coral']
labels = ['Non-fraud', 'Fraud']

plt.figure(figsize=(14, 5))
plt.subplot(121)
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index,
                temp['MAE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.title('Reconstruction MAE')
plt.ylabel('Reconstruction MAE');
plt.xlabel('Index')
plt.subplot(122)
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp.index,
                temp['MSE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.legend(loc=[1, 0], fontsize=12);
plt.title('Reconstruction MSE')
plt.ylabel('Reconstruction MSE');
plt.xlabel('Index')
plt.show()
# 下图分别是MAE和MSE重构误差，其中橘黄色的点是信用欺诈，也就是异常点；蓝色是正常点。我们可以看出异常点的重构误差整体很高。

# 画出Precision-Recall曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i + 1)
    precision, recall, _ = precision_recall_curve(mse_df['Class'], mse_df[metric])
    pr_auc = auc(recall, precision)
    plt.title('Precision-Recall curve based on %s\nAUC = %0.2f' % (metric, pr_auc))
    plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
    plt.xlabel('Recall');
    plt.ylabel('Precision')
plt.show()

# 画出ROC曲线
plt.figure(figsize=(14, 6))
for i, metric in enumerate(['MAE', 'MSE']):
    plt.subplot(1, 2, i + 1)
    fpr, tpr, _ = roc_curve(mse_df['Class'], mse_df[metric])
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic based on %s\nAUC = %0.2f' % (metric, roc_auc))
    plt.plot(fpr, tpr, c='coral', lw=4)
    plt.plot([0, 1], [0, 1], c='dodgerblue', ls='--')
    plt.ylabel('TPR');
    plt.xlabel('FPR')
plt.show()
# 不管是用MAE还是MSE作为划分标准，模型的表现都算是很好的。PR AUC分别是0.51和0.44，而ROC AUC都达到了0.95。

# 画出MSE、MAE散点图
markers = ['o', '^']
colors = ['dodgerblue', 'coral']
labels = ['Non-fraud', 'Fraud']

plt.figure(figsize=(10, 5))
for flag in [1, 0]:
    temp = mse_df[mse_df['Class'] == flag]
    plt.scatter(temp['MAE'],
                temp['MSE'],
                alpha=0.7,
                marker=markers[flag],
                c=colors[flag],
                label=labels[flag])
plt.legend(loc=[1, 0])
plt.ylabel('Reconstruction RMSE');
plt.xlabel('Reconstruction MAE')
plt.show()

# # 使用决策树进行训练模型
# from sklearn import tree
#
# dtc = tree.DecisionTreeClassifier()
# dtc.fit(X_train, y_train)  # 使用训练集进行训练
#
# # 验证模型
# print(dtc.score(X_train, y_train))
# print(dtc.score(X_test, y_test))
# y_predict = dtc.predict(X_test)
#
# from sklearn import metrics
#
# print('Accuracy:', metrics.accuracy_score(y_test, y_predict))
# print('Precision:', metrics.precision_score(y_test, y_predict, average='macro'))
# print('Recall:', metrics.recall_score(y_test, y_predict, average='macro'))
# print('F1-score:', metrics.f1_score(y_test, y_predict, average='macro', zero_division=0))
#
# print()
#
# print('Accuracy:', metrics.accuracy_score(y_test, y_predict))
# print('Precision:', metrics.precision_score(y_test, y_predict, average='micro'))
# print('Recall:', metrics.recall_score(y_test, y_predict, average='micro'))
# print('F1-score:', metrics.f1_score(y_test, y_predict, average='micro', zero_division=0))
#
# print()
#
# print('Accuracy:', metrics.accuracy_score(y_test, y_predict))
# print('Precision:', metrics.precision_score(y_test, y_predict, average='weighted'))
# print('Recall:', metrics.recall_score(y_test, y_predict, average='weighted'))
# print('F1-score:', metrics.f1_score(y_test, y_predict, average='weighted', zero_division=0))
#
# print()
#
# # 可视化决策树
# import pydotplus
#
# with open("tree3.pdf", 'w') as doc_data: dot_data = tree.export_graphviz(dtc, out_file=None,
#                                                                          feature_names=['browsing_time', 'refund_rate',
#                                                                                         'RegisteredTime'],
#                                                                          class_names=['0', '1'],
#                                                                          filled=True, rounded=True,
#                                                                          special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("tree3.pdf")
#
# print(dtc.feature_importances_)
#
# plt.barh(range(4), dtc.feature_importances_, align='center',
#          tick_label=['browsing_time', 'refund_rate', 'RegisteredTime','count'])
# plt.show()
