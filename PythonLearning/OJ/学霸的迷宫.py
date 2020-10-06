a = [[3 for i in range(100)] for j in range(100)]
# 标志是否访问, 0是未访问，1是已经访问
flag = [[0 for i in range(100)] for j in range(100)]
# 方向数组
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]
dz = ['R', 'D', 'L', 'U']
all_dis = []
# 设最短路径长度
min_dis = 99999999
all_dis1 = []
all_str = ''


def dfs(x, y, step):
    if x == p and y == q:
        global min_dis, all_dis, all_str
        # all_dis1.append(all_dis)
        # all_dis = []
        all_dis1.append(all_str)
        all_str = ''
        if step < min_dis:
            min_dis = step

        # 回退
        return

    '''顺时针试探'''
    for i in range(0, 4):
        # global all_dis
        tx = x + dx[i]
        ty = y + dy[i]
        if a[tx][ty] == 0 and flag[tx][ty] == 0:
            all_str += dz[i]
            # all_dis.append(dz[i])
            flag[tx][ty] = 1
            dfs(tx, ty, step + 1)
            flag[tx][ty] = 0

    return


if __name__ == '__main__':
    # 输入m行n列
    m, n = map(int, input().split())
    # 给地图赋值,空地、障碍物
    for i in range(0, m):
        temp = input()
        for j in range(0, n):
            a[i][j] = int(temp[j])

    # 输入起点和终点坐标
    start_x, start_y = 0, 0
    p, q = m - 1, n - 1

    # 从起点开始，所以起点设置为已经访问状态
    flag[start_x][start_y] = 1
    dfs(start_x, start_y, 0)

    print(min_dis)
    print(all_dis1)


'''
3 3
001
100
110
0 0
'''

'''
3 3
000
000
000
'''
