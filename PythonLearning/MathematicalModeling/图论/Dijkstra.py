# dijkstra
from collections import defaultdict
from heapq import *

inf = 99999  # 不连通值


def init_graph(mtx_graph):
    m_n = len(mtx_graph)  # 带权连接矩阵的阶数
    edges = []  # 保存连通的两个点之间的距离(点A、点B、距离)
    for i in range(m_n):
        for j in range(m_n):
            if i != j and mtx_graph[i][j] != inf:
                edges.append((i, j, mtx_graph[i][j]))

    return edges


def dijkstra(edges, from_node, to_node):
    go_path = []
    to_node = to_node - 1
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))
    q, seen = [(0, from_node - 1, ())], set()
    while q:
        (cost, v1, path) = heappop(q)  # 堆弹出当前路径最小成本
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == to_node:
                break
            for c, v2 in g.get(v1, ()):
                if v2 not in seen:
                    heappush(q, (cost + c, v2, path))
    if v1 != to_node:  # 无法到达
        return float['inf'], []

    if len(path) > 0:
        left = path[0]
        go_path.append(left)
        right = path[1]
        while len(right) > 0:
            left = right[0]
            go_path.append(left)
            right = right[1]
        go_path.reverse()  # 逆序变换
        for i in range(len(go_path)):  # 标号加1
            go_path[i] = go_path[i] + 1
    return cost, go_path


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

    edges = init_graph(mtx_graph)
    length, path = dijkstra(edges, 1, 9)
    print('最短距离为：' + str(length))
    print('前进路径为：' + str(path))
