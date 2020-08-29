# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
mpl.rcParams['axes.unicode_minus'] = False
# x_data = np.array([2016, 2017, 2018, 2019])
# y_data = np.array([9.35, 10.0156, 10.87, 11.6])
bar_width = 0.8

x_data = [2016, 2017, 2018, 2019]
y_data = [9.35, 10.0156, 10.87, 11.6]
plt.xlim(2015, 2020)
plt.ylim(0, 12.5)

# plt.bar(x_data, y_data, 0.4, color='red')
plt.bar(x_data, y_data, 0.4)

for x, data in enumerate(y_data):
    plt.text(2016 + x, data + 0.05, data + 0.05, ha='center', va='bottom')


plt.xlabel('年份')
plt.ylabel('GDP/亿万元人民币')
plt.title('近几年粤港澳大湾区GDP情况')
plt.show()