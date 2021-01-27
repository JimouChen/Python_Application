"""
求定积分
"""

import random

if __name__ == '__main__':
    n = int(input('请输入一个较大的整数:'))
    m = 0
    for i in range(n):
        x = random.random()
        y = random.random()
        if y < x ** 2:  # 找到落在f(x)下面的点
            m += 1
    R = m / n
    print(R)
