"""
# @Time    :  2020/11/8
# @Author  :  Jimou Chen
"""
import heapq

Max = 99999999


# 有点像BFS的思想
def dijktestra(graph, start):
    queue = []  # 优先队列
    heapq.heappush(queue, (0, start))
    visited = set()
    path = {start: None}  # 记录该点的上一个点

    # 先把一开始到达的所有路径距离设最大
    distance = {start: 0}
    for vertex in graph:
        if vertex != start:
            distance[vertex] = Max

    while len(queue):
        # 取出的当前在queue的第一个点
        pair = heapq.heappop(queue)
        dist = pair[0]
        vertex = pair[1]
        visited.add(vertex)

        # 该点的所有连接点
        nodes = graph[vertex].keys()
        for v in nodes:
            if v not in visited and dist + graph[vertex][v] < distance[v]:
                heapq.heappush(queue, (dist + graph[vertex][v], v))  # 优先队列会自动把值最小的放在前面
                path[v] = vertex  # 记录上一个点
                distance[v] = dist + graph[vertex][v]  # 更新最小值

    return path, distance


def show_path(path, start, end):
    shortest_path = []
    vertex = end
    while vertex != path[start]:
        vertex = path[vertex]
        shortest_path.append(vertex)

    shortest_path.reverse()
    shortest_path.pop(0)
    shortest_path.append(end)

    return shortest_path


if __name__ == '__main__':
    graph = {
        'A': {'B': 10, 'D': 16, 'I': 5},
        'B': {'A': 10, 'F': 15},
        'C': {'D': 20, 'E': 15, 'I': 6},
        'D': {'A': 16, 'C': 20, 'F': 9},
        'E': {'C': 15, 'H': 4},
        'F': {'B': 15, 'H': 30},
        'G': {'C': 25, 'H': 12},
        'H': {'E': 4, 'F': 9, 'G': 12},
        'I': {'A': 5, 'C': 6}
    }
    path, distance = dijktestra(graph, 'A')
    print('该点的上一个点：', path)
    print('起点到其他各个点的最小距离：', distance)

    shortest_path = show_path(path, 'A', 'H')
    print('shortest_path:', shortest_path)
