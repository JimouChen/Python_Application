"""
# @Time    :  2020/10/6
# @Author  :  Jimou Chen
"""

# 可能会超时

dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]
# min_dis = 99999999


def dfs(x, y, step):
    if x == end_x and y == end_y:
        global min_dis
        if step < min_dis:
            min_dis = step
        return

    for i in range(0, 4):
        tx = x + dx[i]
        ty = y + dy[i]

        if graph[tx][ty] == '.' and flags[tx][ty] == 0:
            flags[tx][ty] = 1
            dfs(tx, ty, step + 1)
            flags[tx][ty] = 0

    return


while True:
    try:
        min_dis = 99999999
        graph = [[0 for i in range(15)] for j in range(15)]
        flags = [[0 for i in range(15)] for j in range(15)]

        n, m, t = map(int, input().split())
        if n == 0 or m == 0:
            break
        start_x, start_y = 0, 0
        end_x, end_y = 0, 0

        for i in range(n):
            string = input()
            for j in range(m):
                graph[i][j] = string[j]
                if string[j] == 'S':
                    start_x, start_y = i, j
                    graph[i][j] = '.'
                if string[j] == 'E':
                    end_x, end_y = i, j
                    graph[i][j] = '.'

        flags[start_x][start_y] = 1
        dfs(start_x, start_y, 0)
        if min_dis < t:
            print('Oh Yes!!!')
        else:
            print('Tragedy!!!')


    except:
        break

'''
4 4 10
....
....
....
S##E


3 4 20
.#E.
.S#.
.#..
'''
