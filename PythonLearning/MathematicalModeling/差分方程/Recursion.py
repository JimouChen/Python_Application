import matplotlib.pyplot as plt
import numpy as np

Max = 665


# 获取相邻纵坐标的差值
def get_delta(y_num: list):
    delta_y = []
    for i in range(len(y_num) - 1):
        delta_y.append(y_num[i + 1] - y_num[i])

    return delta_y


if __name__ == '__main__':
    time = [_ for _ in range(0, 19)]
    number = [9.6, 18.3, 29, 47.2, 71.1, 119.1, 174.6,
              257.3, 350.7, 441.0, 513.3, 559.7, 594.8,
              629.4, 640.8, 651.1, 655.9, 659.6, 661.8]

    plt.title('Relationship between time and number')  # 创建标题
    plt.xlabel('time')  # X轴标签
    plt.ylabel('number')  # Y轴标签
    plt.scatter(time, number)
    plt.plot(time, number)  # 画图
    # plt.show()  # 显示， 注释掉后，实际曲线和预测曲线泛在同一个图里面对比

    delta_p = get_delta(number)
    number.pop(-1)
    pn = np.array(number)
    f = pn * (Max - pn)
    res = np.polyfit(f, delta_p, 1)
    print(res)
    print('k = ', res[0])

    # 预测
    p0 = number[0]
    p_list = []
    for i in range(len(time) + 1):
        p_list.append(p0)
        p0 = res[0] * (Max - p0) * p0 + p0
    plt.xlabel('time')  # X轴标签
    plt.ylabel('number')  # Y轴标签
    plt.title('Prediction')  # 创建标题
    plt.scatter([_ for _ in range(0, len(time) + 1)], p_list, c='r')
    plt.plot(p_list)
    plt.show()
