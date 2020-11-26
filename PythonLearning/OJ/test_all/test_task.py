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
        'A': {'B': 1, 'C': 10, 'D': 6, 'E': 3},
        'B': {'A': 1, 'C': 10, 'F': 10},
        'C': {'A': 10, 'B': 10, 'D': 4, 'F': 1, 'G': 4, 'H': 1},
        'D': {'A': 6, 'C': 4, 'E': 2, 'H': 3},
        'E': {'A': 3, 'D': 2, 'H': 6, 'I': 8},
        'F': {'B': 10, 'C': 1, 'G': 2, 'Z': 5},
        'G': {'F': 2, 'C': 4, 'H': 5, 'Z': 2},
        'H': {'G': 5, 'C': 1, 'D': 3, 'E': 6, 'I': 3, 'Z': 8},
        'I': {'E': 8, 'H': 3, 'Z': 5},
        'Z': {'F': 5, 'G': 2, 'H': 8, 'I': 5}
    }
    path, distance = dijktestra(graph, 'A')
    print(path)
    print(distance)

    shortest_path = show_path(path, 'A', 'Z')
    print('shortest_path:', shortest_path)
