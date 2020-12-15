class Node:
    def __init__(self, pos, val, weight):
        self.pos = pos
        self.val = val
        self.weight = weight


value = [[0 for _ in range(100)] for _ in range(100)]
weight = [[0 for _ in range(100)] for _ in range(100)]
Max = 99999999
res = Node(0, Max, Max)


def bfs():
    global res
    queue = [Node(1, 0, 0)]

    while len(queue):
        temp_node = queue[0]
        queue.pop(0)
        p, v, w = temp_node.pos, temp_node.val, temp_node.weight
        if p == n + 1:
            if (temp_node.weight < res.weight) or (temp_node.weight == res.weight) and (temp_node.val < res.val):
                res = temp_node
            continue

        for i in range(1, m + 1):
            new_val = v + value[p][i]
            new_weight = w + weight[p][i]
            if (new_val > d) or (new_weight > res.weight):
                continue
            queue.append(Node(p + 1, new_val, new_weight))


if __name__ == '__main__':
    # 部件个数，供应商个数，及最大的总价格
    print('请分别输入部件个数，供应商个数，及最大的总价格:')
    n, m, d = map(int, input().split())

    # 各个部件在各个供应商处购买的价格
    print('请输入各个部件在各个供应商处购买的价格:')
    for i in range(1, n + 1):
        temp_list = list(map(int, input().split()))
        for j in range(1, m + 1):
            value[i][j] = temp_list[j - 1]

    # 各个部件在各个供应商处购买的重量
    print('请输入各个部件在各个供应商处购买的重量:')
    for i in range(1, n + 1):
        temp_list = list(map(int, input().split()))
        for j in range(1, m + 1):
            weight[i][j] = temp_list[j - 1]

    bfs()
    print('最小总重量:', res.weight)

'''
4 3 28
9 7 5
10 8 7
5 8 9
4 7 5

3 2 1
2 1 1
1 2 2 
1 2 2


3 3 4
1 2 3
3 2 1
2 2 2
1 2 3
3 2 1
2 2 2

请分别输入部件个数，供应商个数，及最大的总价格:
4 3 21
请输入各个部件在各个供应商处购买的价格:
4 5 6
1 2 3
4 4 6
2 1 3
请输入各个部件在各个供应商处购买的重量:
3 4 5
5 3 2
1 2 3
1 9 2
最小总重量: 7
'''