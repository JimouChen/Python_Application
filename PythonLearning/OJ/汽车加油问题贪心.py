"""
# @Time    :  2020/10/19
# @Author  :  Jimou Chen
"""


def gas(dis: list, n, k):
    for i in dis:
        if i > n:
            print('No Solution')
            return

    s = 0
    num = 0
    for i in range(k + 1):
        s += dis[i]
        if s > n:
            num += 1
            s = dis[i]

    return num


if __name__ == '__main__':
    # 输入加满行驶多远和几个加油站
    n, k = map(int, input().split())
    # 输入从起点到终点中间的k+1个距离
    dis = list(map(int, input().split()))

    count = gas(dis, n, k)
    print(count)

'''
7 7
1 2 3 4 5 1 6 6

4
'''