"""
# @Time    :  2020/11/8
# @Author  :  Jimou Chen
"""


# 深搜
def dfs(graph, start):
    stack = [start]
    visited = set()
    visited.add(start)

    while len(stack):
        vertex = stack.pop()  # 找到栈顶元素
        for v in graph[vertex]:
            if v not in visited:
                stack.append(v)
                visited.add(v)

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

    dfs(graph, 'E')
