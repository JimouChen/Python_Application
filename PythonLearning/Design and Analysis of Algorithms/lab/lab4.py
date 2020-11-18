"""
# @Time    :  2020/11/18
# @Author  :  Jimou Chen
"""


def gas(dis: list, n, m):
    for i in dis:
        if i > n:
            print('汽车无法到达终点')
            return

    s = 0
    num = 0
    gas_num = []  # 记录在哪些汽车站加油
    for i in range(m + 1):
        s += dis[i]
        # 累积距离比汽车可以达到的距离长，则加油
        if s > n:
            num += 1
            # 重置距离
            s = dis[i]
            gas_num.append(i)

    print('最少需要加{}次油'.format(num))
    print('在编号为{}的加油站加过油'.format(gas_num))


if __name__ == '__main__':
    # 输入加满行驶多远和起点到终点有几个加油站
    n, m = map(int, input().split())
    # 输入从起点到终点中间的m+1个距离
    A = list(map(int, input().split()))
    gas(A, n, m)

'''
7 7
1 2 3 4 5 1 6 6

4


4 6
2 3 4 1 2 3 6

无解
'''
