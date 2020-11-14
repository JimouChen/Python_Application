"""
# @Time    :  2020/11/8
# @Author  :  Jimou Chen
"""


# 广搜，start是起点
def bfs(graph, start):
    queue = [start]  # 先把起点入队列
    visited = set()  # 已经访问过的点加入
    visited.add(start)

    while len(queue):
        vertex = queue.pop(0)
        # 找到队列首元素的连接点
        for v in graph[vertex]:
            if v not in visited:
                queue.append(v)
                visited.add(v)
        # 打印弹出队列的该头元素
        print(vertex, end=' ')


if __name__ == '__main__':
    graph = {
        'A': ['B', 'D', 'I'],
        'B': ['A', 'F'],
        'C': ['D', 'E', 'I'],
        'D': ['A', 'C', 'F'],
        'E': ['C', 'H'],
        'F': ['B', 'H'],
        'G': ['C', 'H'],
        'H': ['E', 'F', 'G'],
        'I': ['A', 'C']
    }

    bfs(graph, 'A')
