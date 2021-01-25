import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 计算以p为参数的直线与原始数据之间误差
def f(p):
    k, b = p
    return Y - (k * X + b)


if __name__ == '__main__':
    X = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
    Y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])
    # leastsq使得f的输出数组的平方和最小，参数初始值为[1,0]
    r = leastsq(f, [1, 0])  # 数初始值可以随便设个合理的
    k, b = r[0]
    x = np.linspace(0, 10, 1000)
    y = k * x + b

    # 画散点图，s是点的大小
    plt.scatter(X, Y, s=100, alpha=1.0, marker='o', label=u'数据点')
    # 话拟合曲线，linewidth是线宽
    plt.plot(x, y, color='r', linewidth=2, linestyle="-", markersize=20, label=u'拟合曲线')
    plt.xlabel('安培/A')  # 美赛就不用中文了
    plt.ylabel('伏特/V')
    plt.legend(loc=0, numpoints=1)  # 显示点和线的说明
    # plt.plot(X, Y)
    plt.show()

    print('k = ', k)
    print('b = ', b)
