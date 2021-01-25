from scipy.optimize import minimize
import numpy as np


# f = 1/x+x
def fun(args):
    a = args
    return lambda x: a / x[0] + x[0]


if __name__ == '__main__':
    args = (1)  # a
    # x0 = np.asarray((1.5))  # 初始猜测值
    # x0 = np.asarray((2.2))  # 初始猜测值
    x0 = np.asarray((2))  # 设置初始猜测值

    res = minimize(fun(args), x0, method='SLSQP')
    print('最值:', res.fun)
    print('是否是最优解', res.success)
    print('取到最值时，x的值(最优解)是', res.x)
