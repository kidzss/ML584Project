## 584 Project
### browsing_times.csv 浏览时长
- 浏览时间长度都很短 10s之内，正常用户30s-60s
- 注册时间和下单时间是同一天并且很紧

```
	if data.loc[x, 'user_id'] in feature_users:
      t = r.randint(5, 30)  #特征用户， 单位是秒
   else:
      t = r.randint(30, 80)  #非特征用户 单位是秒
```

### feature_users.csv 特征用户
- 账号，注册时间(早于第一次下单时间)
- 第一次下单时间减去t = r.randint(20, 100) 20分钟到100分钟的随机数字

### normal_users.csv 普通用户
- 账号，注册时间(早于第一次下单时间)
- 第一次下单时间减去t = r.randint(60, 129600) 1小时到90天的随机数字

### flight_number_tab.csv 航班表
### flight_refund_rate_tab.csv 航班退频率
### main.py 执行一些就可以从新获取上面的表
### orders.csv 原来的订单表，经过修改了，变成了11个航班