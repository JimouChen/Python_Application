import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import pandas as pd


# 计算以p为参数的直线与原始数据之间误差
def f(p):
    k, b = p
    return Y - (k * X + b)


if __name__ == '__main__':
    em_data = pd.read_csv('data/em_data.csv')
    extension_rate = em_data.iloc[:45, 0]
    mass_loss = em_data.iloc[:45, 1]
    # extension_rate = em_data.iloc[:, 0]
    # mass_loss = em_data.iloc[:, 1]
    # X = extension_rate
    # Y = mass_loss
    X = extension_rate.apply(np.log)
    Y = mass_loss.apply(np.log)
    # leastsq使得f的输出数组的平方和最小，参数初始值为[1,0]
    r = leastsq(f, [1, 0])  # 数初始值可以随便设个合理的
    k, b = r[0]
    x = np.linspace(-10, 10, 1000)
    y = k * x + b

    plt.figure(12)
    plt.subplot(121)
    # 画散点图，s是点的大小
    plt.scatter(X, Y, s=30, alpha=1.0, marker='o', label='data points')
    # 话拟合曲线，linewidth是线宽
    plt.plot(x, y, color='r', linewidth=2, linestyle="-", markersize=20, label='Curve fitting')
    plt.xlabel('log(Extension rate mm day^(-1))')
    plt.ylabel('log(Decomposition rate %mass loss)')
    # plt.title('log(Decomposition rate) ~ log(Extension rate)')
    plt.xlim(-2.5, 7)
    plt.ylim(2, 5)
    plt.legend(loc=0, numpoints=1)  # 显示点和线的说明

    # plt.show()

    print('k = ', k)
    print('b = ', b)

    # ds = np.e
    ds = np.e

    plt.subplot(122)
    plt.xlabel('Extension rate mm day^(-1)')
    plt.ylabel('Decomposition rate %mass loss')
    # plt.title('Decomposition rate ~ Extension rate')

    plt.scatter(extension_rate, mass_loss, s=50, alpha=1.0, marker='o', label='data points')
    ex = np.array([_ for _ in range(1, 16)])
    plt.plot(ex, (ds ** b) * ex ** k, color='r', linewidth=2, linestyle="-", markersize=20, label='Curve fitting')
    plt.legend(loc=0, numpoints=1)  # 显示点和线的说明
    plt.show()