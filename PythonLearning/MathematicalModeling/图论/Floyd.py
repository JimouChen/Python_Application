import numpy as np

inf = 99999  # 不连通值


def floyd(graph):
    N = len(graph)
    A = np.array(graph)
    path = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            if A[i][j] != inf:
                path[i][j] = j

    for k in range(0, N):
        for i in range(0, N):
            for j in range(0, N):
                if A[i][k] + A[k][j] < A[i][j]:
                    A[i][j] = A[i][k] + A[k][j]
                    path[i][j] = path[i][k]

    for i in range(0, N):
        for j in range(0, N):
            path[i][j] = path[i][j] + 1

    print('距离 = \n', A)
    print('路径 = \n', path)


if __name__ == '__main__':
    mtx_graph = [[0, 1, inf, 3, inf, inf, inf, inf, inf],
                 [1, 0, 5, inf, 2, inf, inf, inf, inf],
                 [inf, inf, 0, 1, inf, 6, inf, inf, inf],
                 [inf, inf, inf, 0, inf, 7, inf, 9, inf],
                 [inf, 2, 3, inf, 0, 4, 2, inf, 8],
                 [inf, inf, 6, 7, inf, 0, inf, 2, inf],
                 [inf, inf, inf, inf, inf, 1, 0, inf, 3],
                 [inf, inf, inf, inf, inf, inf, 1, 0, 2],
                 [inf, inf, inf, inf, 8, inf, 3, 2, 0]]
    floyd(mtx_graph)
