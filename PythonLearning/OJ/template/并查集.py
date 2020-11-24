"""
# @Time    :  2020/11/24
# @Author  :  Jimou Chen
"""
Max = 1000
parents = [i for i in range(1, Max)]  # 初始化Max个顶点


# 查询
def find(x):
    if parents[x] == x:
        return x
    else:
        t = find(parents[x])  # 优化
        parents[x] = t
        return t


# 合并
def union(x, y):
    x = find(x)
    y = find(y)
    if x == y:
        return
    parents[x] = y


if __name__ == '__main__':
    pass
