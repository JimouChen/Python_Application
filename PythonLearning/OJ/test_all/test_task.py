"""
# @Time    :  2020/10/6
# @Author  :  Jimou Chen
"""


class Point:
    def __init__(self, x, y, step):
        self.x = x
        self.y = y
        self.step = step


dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]
graph = [[0 for i in range(11)] for j in range(11)]
flags = [[0 for i in range(11)] for j in range(11)]
flag = 0

while True:
    try:
        queue = []
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

        start_point = Point(start_x, start_y, 0)
        queue.append(start_point)
        flags[start_x][start_y] = 1

        while len(queue) != 0:
            x = queue[0].x
            y = queue[0].y
            if x == end_x and y == end_y:
                flag = 1
                print('step = ', queue[0].step)
                break

            for k in range(0, 4):
                tx = x + dx[k]
                ty = y + dy[k]
                if graph[tx][ty] == '.' and flags[tx][ty] == 0:
                    # 入队
                    temp = Point(tx, ty, queue[0].step + 1)
                    queue.append(temp)
                    flags[tx][ty] = 1

            queue.pop(0)

        if flag == 0:
            print('no ans')
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
