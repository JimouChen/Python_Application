import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    data = np.genfromtxt('data/xyz_data.csv', delimiter=',')
    x_data = np.log(np.abs(data[1:, :-1]*20))
    y_data = np.log(np.abs(data[1:, 2]))

    model = LinearRegression()
    model.fit(x_data, y_data)
    print('coefficient:', model.coef_)
    print('intercept:', model.intercept_)

    # plt.figure(12)
    # plt.subplot(121)

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=30)
    x0 = x_data[:, 0]
    x1 = x_data[:, 1]
    # 生成网格矩阵
    x0, x1 = np.meshgrid(x0, x1)
    z = model.intercept_ + x0 * model.coef_[0] + x1 * model.coef_[1]
    # 画3D图
    ax.plot_surface(x0, x1, z)
    # 设置坐标轴
    ax.set_xlabel('log(Tolerance to moisture /MPa)')
    ax.set_ylabel('log(Extension rate /mm day^(-1))')
    ax.set_zlabel('log(Decomposition rate /%)')
    # 显示图像
    plt.show()

    ax = plt.figure().add_subplot(111, projection='3d')
    x_data = data[1:, :-1]
    x0 = x_data[:, 0]
    x1 = x_data[:, 1]
    ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=30)  # 点为红色三角形
    x0, x1 = np.meshgrid(x0, x1)
    z = np.e ** model.intercept_ * x0 ** model.coef_[0] * x1 ** model.coef_[1]
    ax.plot_surface(x0, x1, z)
    ax.set_xlabel('Tolerance to moisture /MPa')
    ax.set_ylabel('Extension rate /mm day^(-1)')
    ax.set_zlabel('Decomposition rate /%')
    plt.show()
