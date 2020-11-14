"""
# @Time    :  2020/10/6
# @Author  :  Jimou Chen
"""


class Point:
    def __init__(self, x, y, step):
        self.x = x
        self.y = y
        self.step = step


queue = []
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]
graph = [[0 for i in range(100)] for j in range(100)]
flags = [[0 for i in range(100)] for j in range(100)]
flag = 0

if __name__ == '__main__':
    m, n = map(int, input().split())
    for i in range(m):
        l = list(map(int, input().split()))
        for j in range(n):
            graph[i][j] = l[j]

    start_x, start_y = map(int, input().split())
    end_x, end_y = map(int, input().split())

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
            if graph[tx][ty] == 1 and flags[tx][ty] == 0:
                # 入队
                temp = Point(tx, ty, queue[0].step + 1)
                queue.append(temp)
                flags[tx][ty] = 1

        queue.pop(0)

    if flag == 0:
        print('no ans')

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
