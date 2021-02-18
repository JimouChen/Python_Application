# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from mpl_toolkits.mplot3d import Axes3D
#
# if __name__ == '__main__':
#     data = np.genfromtxt('data/Fungi_temperature_curves.csv', delimiter=',')
#     x_data = data[1:5502, :-1]
#     y_data = data[1:5502, -1]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures  # 生成多项式用的

if __name__ == '__main__':
    # 读取数据
    data = np.genfromtxt(r'data/Fungi_temperature_curves.csv', delimiter=',')
    num = [1, 5502, 11003, 16504, 22005, 27506, 33007, 38508, 44009, 49510,
           55011, 60512, 66013, 71514, 77015]

    num2 = [143027, 148528, 154029,
            159530, 165031, 170532]

    color = ['black', 'peru', 'blue', 'brown', 'darkcyan', 'red', 'teal', 'green',
             'yellow', 'hotpink', 'darkblue', 'cyan', 'magenta', 'maroon', 'purple',
             'orange', 'orangered', 'gold', 'pink', 'skyblue', 'springgreen']

    plt.figure(21)
    plt.subplot(211)

    for i in range(len(num) - 1):
        x_data = data[num[i]:num[i + 1], 1]
        y_data = data[num[i]:num[i + 1], 2]

        # 转换为二维数据
        x_data = x_data[:, np.newaxis]  # 或者x_data = data[1:, 1, np.newaxis]
        y_data = y_data[:, np.newaxis]  # 或者y_data = data[1:, -1, np.newaxis]

        # 画图
        plt.plot(x_data, y_data, '-', color=color[i], markersize=1)

    for i in range(len(num2) - 1):
        x_data = data[num2[i]:num2[i + 1], 1]
        y_data = data[num2[i]:num2[i + 1], 2]

        # 转换为二维数据
        x_data = x_data[:, np.newaxis]  # 或者x_data = data[1:, 1, np.newaxis]
        y_data = y_data[:, np.newaxis]  # 或者y_data = data[1:, -1, np.newaxis]

        # 画图
        plt.plot(x_data, y_data, '-', color=color[i + 14], markersize=1)

    plt.title('The relationship between temperature and extension rate of 20 fungi')
    plt.xlabel('Temperature(℃)')
    plt.ylabel('Extension rate(mm/day)')
    # plt.show()

    plt.subplot(212)
    data = np.genfromtxt(r'data/Fungi_moisture_curves.csv', delimiter=',')
    num3 = [1, 502, 1003, 1504, 2005, 2506, 3007, 3508, 4009, 4510,
            5011, 5512, 6013, 6514, 7015, 7516, 8017, 8518, 9019, 9520,
            10021]

    for i in range(len(num3) - 1):
        x_data = data[num3[i]:num3[i + 1], 1]
        y_data = data[num3[i]:num3[i + 1], 2]

        # 转换为二维数据
        x_data = x_data[:, np.newaxis]  # 或者x_data = data[1:, 1, np.newaxis]
        y_data = y_data[:, np.newaxis]  # 或者y_data = data[1:, -1, np.newaxis]

        # 画图
        plt.plot(x_data, y_data, '-', color=color[i], markersize=1)

    plt.title('The relationship between moisture and extension rate of 20 fungi')
    plt.xlabel('moisture(MPa)')
    plt.ylabel('Extension rate(mm/day)')
    plt.show()

