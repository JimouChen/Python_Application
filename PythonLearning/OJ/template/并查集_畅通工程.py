"""
# @Time    :  2020/11/24
# @Author  :  Jimou Chen
"""


# 查询
def find(x):
    if par[x] == x:
        return x
    else:
        t = find(par[x])  # 优化
        par[x] = t
        return t


# def find(x):
#     if par[x] != x:
#         t = find(par[x])
#         par[x] = t
#         return t
#     return x


def union(x, y):
    x = find(x)
    y = find(y)
    if x == y:
        return 0
    par[x] = y
    return 1


while True:
    try:
        # n个节点，m个集合
        n, m = map(int, input().split())
        if n == 0:
            break
        par = [i for i in range(n)]

        # 若合并到最后有cnt个集合，那么也就是说至少需要cnt-1条边使得任意一个点可以通畅到达其他任意一个点
        for i in range(m):
            a, b = map(int, input().split())
            union(a - 1, b - 1)

        cnt = 0
        for i in range(0, n):
            if par[i] == i:
                cnt += 1
        print(cnt - 1)

    except:
        break

'''
5 3
1 2
3 2
4 5

1

4 2
1 3
4 3

1

0


'''
