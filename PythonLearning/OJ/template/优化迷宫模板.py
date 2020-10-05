"""
# @Time    :  2020/10/5
# @Author  :  Jimou Chen
"""
# 迷宫二维列表 n行m列 ：[[0 for i in range(m)] for j in range(n)]
# 假设1 是空地，2 是障碍物
a = [[0 for i in range(100)] for j in range(100)]
# 标志是否访问, 0是未访问，1是已经访问
flag = [[0 for i in range(100)] for j in range(100)]
# 方向数组
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

# 设最短路径长度
min_dis = 99999999


def dfs(x, y, step):
    if x == p and y == q:
        global min_dis
        if step < min_dis:
            min_dis = step
        # 回退
        return

    '''顺时针试探'''
    for i in range(0, 3):
        tx = x + dx[i]
        ty = y + dy[i]
        if a[tx][ty] == 1 and flag[tx][ty] == 0:
            flag[tx][ty] = 1
            dfs(tx, ty, step + 1)
            flag[tx][ty] = 0

    return


if __name__ == '__main__':
    # 输入m行n列
    m, n = map(int, input().split())
    # 给地图赋值,空地、障碍物
    for i in range(0, m):
        temp = list(map(int, input().split()))
        for j in range(0, n):
            # a[i][j] = int(input())
            a[i][j] = temp[j]

    # 输入起点和终点坐标
    start_x, start_y = map(int, input().split())
    # global p, q
    p, q = map(int, input().split())

    # 从起点开始，所以起点设置为已经访问状态
    flag[start_x][start_y] = 1
    dfs(start_x, start_y, 0)

    print(min_dis)

'''
5 4
1 1 2 1
1 1 1 1
1 1 2 1
1 2 1 1
1 1 1 2
0 0
3 2
'''
