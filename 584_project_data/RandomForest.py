import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据初始化
orders = pd.read_csv("orders_all_join_tab.csv", keep_default_na=False, na_filter=True)
devices = pd.read_csv('device_repeat_tab.csv', keep_default_na=False, na_filter=True)

devices_s_df = devices[['device_id', 'repeat']]
ret_df = pd.merge(orders, devices_s_df, how='left', on='device_id')

orders = ret_df.drop(columns=['Unnamed: 0', 'device_id'], axis=1)

orders = orders.dropna()
orders.head()

# 计算注册时间和下单时间差值
orders['ordermins'] = pd.to_datetime(orders['order_date'])
orders['joinmins'] = pd.to_datetime(orders['join_date'])
orders['RegisteredTime'] = (orders['ordermins'] - orders['joinmins']).dt.seconds
# 如果条件为真，返回真 否则返回假
# condition_is_true if condition else condition_is_false
certificate_len = []
for x in orders.index:
    if len(orders.loc[x, 'certificate']) > 9:
        certificate_len.append(0)
    else:
        certificate_len.append(1)

orders['certificate_len'] = certificate_len
orders = orders.drop(columns=['certificate'], axis=1)

feature = orders[['browsing_time', 'refund_rate', 'RegisteredTime', 'count', 'repeat', 'certificate_len', 'flag']]
feature_values = orders[['browsing_time', 'refund_rate', 'RegisteredTime', 'count', 'flag']].values
feature_pre = feature

sns.pairplot(feature, hue='flag')
plt.show()

X = pd.get_dummies(feature.drop('flag', axis=1), drop_first=True)
y = feature['flag']
print(X.head())

# 训练
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
model = RandomForestClassifier(n_estimators=7, max_features='auto', random_state=101)
model.fit(X_train, y_train)

# 预测
from sklearn.metrics import accuracy_score

preds = model.predict(X_test)
print(accuracy_score(y_test, preds))

from sklearn import metrics

print('Accuracy:', metrics.accuracy_score(y_test, preds))
print('Precision:', metrics.precision_score(y_test, preds, average='weighted'))
print('Recall:', metrics.recall_score(y_test, preds, average='weighted'))
print('F1-score:', metrics.f1_score(y_test, preds, average='weighted', zero_division=0))
