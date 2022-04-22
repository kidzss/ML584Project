import random

import pandas as pd
import random as r
from datetime import *
import numpy as np


# save dataframe to csv file
def write_csv_file(file_name, data):
    df = pd.DataFrame(data)
    df.to_csv(file_name + '.csv')


# read file from csv file
def read_csv_file(file_name):
    df = pd.read_csv(file_name + '.csv')
    return df


# 生成随机字符串
def random_str(num):
    # 猜猜变量名为啥叫 H
    H = 'abcdefghijklmnopqrstuvwxyz0123456789'
    salt = ''
    for i in range(num):
        salt += r.choice(H)

    return salt


# 生成随机的device id
def random_device_id(num):
    ret = ''
    for i in range(num):
        if len(ret):
            ret += '-'
        ret += random_str(5)
    return ret


# 生成随机的特征用户的device id，一共有5个，给30个用户重复使用
def random_feature_users_devceid(num):
    ret = []
    for i in range(num):
        ret.append(random_device_id(5))
    return ret


# 生成特征用户
def get_feature_users_tab(data):
    # random 30 user_id
    count_row = data.shape[0]
    feature_users = []
    for i in range(30):
        j = r.randint(0, count_row - 1)
        if data.loc[j, 'user_id'] not in feature_users:
            feature_users.append(data.loc[j, 'user_id'])
        else:
            i -= 1

    # list 去重
    feature_users = list(set(feature_users))

    # 找到随机到的feature_users 的注册时间最早的日期
    join_date_map = {}
    for i in range(len(feature_users)):
        tmp_data = datetime.now()
        if feature_users[i] in join_date_map:
            tmp_data = join_date_map[feature_users[i]]

        for x in data.index:
            if data.loc[x, 'user_id'] == feature_users[i]:
                # print(data.loc[x, 'order_date'])
                b = datetime.strptime(data.loc[x, 'order_date'], '%Y/%m/%d %H:%M')
                if tmp_data > b:
                    join_date_map[feature_users[i]] = b
    # print()
    # print(join_date_map)

    join_date_list = []
    values = np.random.normal(loc=60, scale=10.0, size=30).astype(int)
    print('----特征用户加入日期减去值')
    print(values)
    print()

    for i in range(len(feature_users)):
        key = feature_users[i]
        if key in join_date_map:
            t = r.randint(0, 29)
            t_v = int(values[t])
            datetime2 = join_date_map[key] - timedelta(minutes=t_v)  # 减t分钟
            join_date_list.append(datetime2)

    features = {'user_id': feature_users, 'join_date': join_date_list, 'flag': np.ones(len(feature_users), dtype=int)}
    write_csv_file('feature_users', features)
    return feature_users


# 数据初始化
def init_orders():
    data = read_csv_file('orders')
    # remove duplicate rows
    data.drop_duplicates(inplace=True)

    # change order_status from T to R
    for x in data.index:
        if data.loc[x, 'order_status'] == 'T':
            data.loc[x, 'order_status'] = 'R'
    return data


# 生成浏览时长
def get_browsing_times_tab(orders_df, feature_users):
    # 生成浏览时长
    data = orders_df[['user_id', 'order_id']]
    orders_id = []
    browsing_time = []
    users_id = []

    f_b_values = np.random.normal(loc=15, scale=4.0, size=30).astype(int)
    b_values = np.random.normal(loc=100, scale=25.0, size=1200).astype(int)
    print('----特征用户browsing_time')
    print(f_b_values)
    print(b_values)
    print()
    for x in data.index:
        t_v = -1
        if data.loc[x, 'user_id'] in feature_users:
            t = r.randint(1, 30)  # 单位是秒
            t_v = b_values[t]
        else:
            t = r.randint(15, 180)  # 单位是秒
            t_v = b_values[t]

        browsing_time.append(int(t_v))
        orders_id.append(data.loc[x, 'order_id'])
        users_id.append(data.loc[x, 'user_id'])

    order_browsing_times = {'user_id': users_id, 'order_id': orders_id, 'browsing_time': browsing_time}
    write_csv_file('browsing_times', order_browsing_times)

    return order_browsing_times


# 生成航班表
def get_flight_num_tab(orders_df):
    data = orders_df[
        ['flight_number', 'flight_time', 'flight_num_date', 'dport', 'aport', 'dport_city', 'aport_city', 'dcity_code',
         'acity_code']]
    print()
    flight_data = data.drop_duplicates()
    write_csv_file('flight_number_tab', flight_data)

    # print(data[data.duplicated(keep=False)])
    return flight_data


# 当前航班的退频率
def get_refund_rate_tab(orders_df, flight_nums_data):
    # 航班退票率的表
    order_data = orders_df[['flight_num_date', 'order_status']]
    refund_rate = {}

    for i in flight_nums_data.index:
        key = flight_nums_data.loc[i, 'flight_num_date']
        t_count = 0
        c_count = 0
        for j in order_data.index:
            if order_data.loc[j, 'flight_num_date'] == key:
                val = order_data.loc[j, 'order_status']
                if val == 'R':
                    t_count += 1
                c_count += 1
        rate = 0.0
        if c_count > 0:
            rate = t_count / c_count
        refund_rate[key] = format(rate, '.4f')

    keys = list(refund_rate.keys())
    values = list(refund_rate.values())
    refund_rate_tab = {'flight_num_date': keys, 'refund_rate': values}
    write_csv_file('flight_refund_rate_tab', refund_rate_tab)

    # 表的合并
    refund_rate_tab_df = read_csv_file('flight_refund_rate_tab')
    flight_nums_data_df = read_csv_file('flight_number_tab')
    ret_df = pd.merge(refund_rate_tab_df, flight_nums_data_df, how='right', on='flight_num_date')
    new_df = ret_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)
    new_df = new_df[
        ['flight_number', 'flight_time', 'flight_num_date', 'refund_rate', 'dport', 'aport', 'dport_city', 'aport_city',
         'dcity_code', 'acity_code']]
    write_csv_file('flight_refund_rate_tab', new_df)


# 获取普通的注册用户
def get_normal_users(orders_df, feature_users):
    normal_users = []
    count_row = orders_df.shape[0]
    for x in orders_df.index:
        if orders_df.loc[x, 'user_id'] not in feature_users:
            normal_users.append(orders_df.loc[x, 'user_id'])

    # list 去重
    normal_users = list(set(normal_users))

    # 找到随机到的normal_users 的注册时间最早的日期
    join_date_map = {}
    for i in range(len(normal_users)):
        tmp_data = datetime.now()
        if normal_users[i] in join_date_map:
            tmp_data = join_date_map[normal_users[i]]

        for x in orders_df.index:
            if orders_df.loc[x, 'user_id'] == normal_users[i]:
                # print(data.loc[x, 'order_date'])
                b = datetime.strptime(orders_df.loc[x, 'order_date'], '%Y/%m/%d %H:%M')
                if tmp_data > b:
                    join_date_map[normal_users[i]] = b

    join_date_list = []
    for i in range(len(normal_users)):
        key = normal_users[i]
        if key in join_date_map:
            t_list = np.random.normal(loc=1080, scale=280.0, size=1200).astype(int)
            t_i = r.randint(0, len(normal_users) - 1)
            datetime2 = join_date_map[key] - timedelta(hours=int(t_list[t_i]))  # 减t分钟
            join_date_list.append(datetime2)

    features = {'user_id': normal_users, 'join_date': join_date_list,
                'flag': np.zeros(len(normal_users), dtype=int)}
    write_csv_file('normal_users', features)

    # 表的合并
    normal_users_df = read_csv_file('normal_users')
    feature_users_df = read_csv_file('feature_users')

    # 现将表构成list，然后在作为concat的输入
    frames = [normal_users_df, feature_users_df]
    result = pd.concat(frames, ignore_index=True)
    new_df = result.drop(columns=['Unnamed: 0'], axis=1)
    user_register = new_df.drop_duplicates()

    write_csv_file('user_register', user_register)

    # user_register.csv
    return normal_users


# 生成每个订单的device id
def get_order_devices_id(orders_df, feature_users_list):
    data = orders_df[['user_id', 'order_id']]
    orders_id_devices = {}
    browsing_time = []
    users_id = []
    fetaure_device_list = random_feature_users_devceid(5)
    print()
    print('特征设备列表%s' % fetaure_device_list)
    for x in data.index:
        t = -1
        if data.loc[x, 'user_id'] in feature_users_list:
            t = r.randint(0, 4)
            device_id = fetaure_device_list[t]
        else:
            device_id = random_device_id(5)

        orders_id_devices[data.loc[x, 'order_id']] = device_id

    keys = list(orders_id_devices.keys())
    values = list(orders_id_devices.values())
    order_device_id = {'order_id': keys, 'device_id': values}
    write_csv_file('order_device_id', order_device_id)

    # 表的合并
    order_device_df = read_csv_file('order_device_id')
    orders_data_df = read_csv_file('orders')
    ret_df = pd.merge(order_device_df, orders_data_df, how='right', on='order_id')

    new_df = ret_df.drop(columns=['Unnamed: 0'], axis=1)
    # print(new_df.to_string())
    new_df = new_df[
        ['user_id', 'order_id', 'order_date', 'booking_date', 'order_status', 'device_id', 'dport', 'aport',
         'dport_city', 'aport_city', 'dcity_code', 'acity_code', 'flight_time', 'flight_number', 'flight_date',
         'flight_num_date']]
    write_csv_file('orders_add_device_tab', new_df)


# 生成护照
import string
def certificate_init(pre_str, len):
    ret = pre_str
    for i in range(len):
        ret += str(random.randint(0, 9))
    return ret


if __name__ == '__main__':
    # 数据初始化
    orders_df = init_orders()

    # 生成特征用户
    feature_users_list = get_feature_users_tab(orders_df)

    # 生成普通用户
    normal_users_list = get_normal_users(orders_df, feature_users_list)

    # 生成浏览时长
    browsing_times = get_browsing_times_tab(orders_df, feature_users_list)

    # 生成航班信息
    flight_nums_data = get_flight_num_tab(orders_df)

    # 当天航班的退频率
    get_refund_rate_tab(orders_df, flight_nums_data)

    # 生成订单 device id
    get_order_devices_id(orders_df, feature_users_list)

    # 表的合并
    orders_add_device_tab_df = read_csv_file('orders_add_device_tab')
    browsing_times_df = read_csv_file('browsing_times')
    browsing_times_df = browsing_times_df[['order_id', 'browsing_time']]
    ret_df = pd.merge(orders_add_device_tab_df, browsing_times_df, how='inner', on='order_id')

    new_df = ret_df.drop(columns=['Unnamed: 0'], axis=1)

    write_csv_file('orders_new_tab', new_df)

    # 表的合并
    orders_new_tab_df = read_csv_file('orders_new_tab')
    flight_refund_rate_df = read_csv_file('flight_refund_rate_tab')
    flight_refund_rate_df = flight_refund_rate_df[['flight_num_date', 'refund_rate']]
    ret_df = pd.merge(orders_new_tab_df, flight_refund_rate_df, how='left', on='flight_num_date')

    # print(ret_df.to_string())
    new_df = ret_df.drop(columns=['Unnamed: 0'], axis=1)
    write_csv_file('orders_all_tab', new_df)

    # 用户购买次数
    browsing_times_df = read_csv_file('browsing_times')
    buy_count_s = browsing_times_df['user_id'].value_counts()
    # buy_count_df = buy_count_s.to_frame()

    buy_count_se = {'user_id': buy_count_s.index, 'count': buy_count_s.values}
    write_csv_file('user_buy_count_tab', buy_count_se)

    user_register_df = read_csv_file('user_register')
    user_register_df = user_register_df[['user_id', 'join_date', 'flag']]

    user_buy_count_df = read_csv_file('user_buy_count_tab')
    user_buy_count_df = user_buy_count_df[['user_id', 'count']]

    ret_df = pd.merge(user_register_df, user_buy_count_df, how='inner', on='user_id')

    user_register = ret_df.drop_duplicates()
    write_csv_file('user_register', user_register)

    # 生成护照
    user_register_df = read_csv_file('user_register')
    normal_users_df = read_csv_file('normal_users')

    nor_sample_df = normal_users_df.sample(n=20, replace=False, axis=0)

    values = nor_sample_df['user_id'].drop_duplicates().values.tolist()

    certificate_list = []
    for x in user_register_df.index:
        if user_register_df.loc[x, 'flag'] == 0:
            if user_register_df.loc[x, 'user_id'] not in values:
                certificate = certificate_init('G', 9)
                certificate_list.append(certificate)
            else:
                certificate = certificate_init('G', 5)
                certificate_list.append(certificate)
        else:
            certificate = certificate_init('G', 9)
            certificate_list.append(certificate)

    user_register_df['certificate'] = certificate_list
    new_df = user_register_df.drop(columns=['Unnamed: 0'], axis=1)
    user_register = new_df.drop_duplicates()

    write_csv_file('user_register', user_register)

    # 生成设备重复次数
    orders_all_tab_df = read_csv_file('orders_all_tab')
    device_repeat_s = orders_all_tab_df['device_id'].value_counts()

    buy_count_se = {'device_id': device_repeat_s.index, 'repeat': device_repeat_s.values}
    write_csv_file('device_repeat_tab', buy_count_se)

    # 合并注册时间和购买次数和护照
    orders_all_tab_df = read_csv_file('orders_all_tab')

    for x in orders_all_tab_df.index:
        if orders_all_tab_df.loc[x, 'user_id'] in feature_users_list:
            orders_all_tab_df.loc[x, 'order_status'] = 'R'

    user_register_df = read_csv_file('user_register')
    user_register_df = user_register_df[['user_id', 'join_date', 'count', 'certificate', 'flag']]
    ret_df = pd.merge(orders_all_tab_df, user_register_df, how='left', on='user_id')

    new_df = ret_df.drop(columns=['Unnamed: 0'], axis=1)
    write_csv_file('orders_all_join_tab', new_df)
