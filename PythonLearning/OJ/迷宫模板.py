"""
# @Time    :  2020/9/30
# @Author  :  Jimou Chen
"""
dx = [-1, 0, 1, 0]
dy = [0, -1, 0, 1]


def f(x, y):
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
    f(nx, ny)
