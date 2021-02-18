import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import pandas as pd


# 计算以p为参数的直线与原始数据之间误差
def f(p):
    k, b = p
    return Y - (k * X + b)


if __name__ == '__main__':
    data = np.genfromtxt('data.csv', delimiter=',')
    X = data[:, 0]
    Y = data[:, -1]

    # 画散点图，s是点的大小
    plt.scatter(X, Y, s=30, alpha=1.0, marker='o', label='Test points')
    # 话拟合曲线，linewidth是线宽
    plt.plot(X, Y, color='r', linewidth=2, linestyle="-", markersize=20, label='Trend curve')
    plt.xlabel(' Tolerance to moisture')
    plt.ylabel('Decomposition rate %mass loss')
    plt.title(' Sensitivity of tolerance to moisture')
    # plt.xlim(-2.5, 7)
    # plt.ylim(2, 5)
    plt.legend(loc=0, numpoints=1)  # 显示点和线的说明

    plt.show()
