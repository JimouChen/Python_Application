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
graph = [[3 for i in range(100)] for j in range(100)]
flags = [[0 for i in range(100)] for j in range(100)]
dz = ['R', 'D', 'L', 'U']
flag = 0

if __name__ == '__main__':
    m, n = map(int, input().split())
    for i in range(m):
        # l = list(map(int, input().split()))
        string = input()
        for j in range(n):
            graph[i][j] = int(string[j])

    start_x, start_y = 0, 0
    end_x, end_y = m - 1, n - 1

    start_point = Point(start_x, start_y, 0)
    queue.append(start_point)
    flags[start_x][start_y] = 1

    while len(queue) != 0:
        x = queue[0].x
        y = queue[0].y
        if x == end_x and y == end_y:
            flag = 1
            print('step =', queue[0].step)
            break

        for k in range(0, 4):
            tx = x + dx[k]
            ty = y + dy[k]
            if graph[tx][ty] == 0 and flags[tx][ty] == 0:
                # 入队
                temp = Point(tx, ty, queue[0].step + 1)
                queue.append(temp)
                flags[tx][ty] = 1
                # 打印轨迹
                print(dz[k])

        queue.pop(0)

    if flag == 0:
        print('no ans')

'''
3 3
000
000
000
'''
