"""
# @Time    :  2020/11/26
# @Author  :  Jimou Chen
"""
'''
输入n个顶点，m种颜色，还有该图的邻接矩阵
假设第一个顶点是1，第一种颜色是1
'''
n, m = map(int, input().split())
graph = [[0 for _ in range(100)] for _ in range(100)]  # 存放邻接矩阵
color = [0 for _ in range(100)]  # 存放最后符合着色情况
cnt = 0  # 记录有多少种着色方案


# 检查第i个顶点的颜色是否满足条件
def check(k):
    for i in range(1, k + 1):
        # k与i之间相连并且i顶点的颜色与k顶点的颜色相同
        if graph[k][i] == 1 and color[i] == color[k]:
            return 0
    return 1


def dfs(step):
    global cnt
    # 所有的顶点已经涂完颜色
    if step == n + 1:
        for i in range(1, n + 1):
            print(color[i], end=' ')
        print()
        cnt += 1
        return

    # 遍历填m种颜色
    for i in range(1, m + 1):
        color[step] = i
        if check(step):
            dfs(step + 1)
        color[step] = 0  # 回溯，0表示没有着色


if __name__ == '__main__':
    for i in range(1, n + 1):
        temp = list(map(int, input().split()))
        for j in range(1, n + 1):
            graph[i][j] = temp[j - 1]

    dfs(1)
    print('总方案数: ', cnt)

'''

5 4 
0 1 1 1 0 
1 0 1 1 1 
1 1 0 1 0 
1 1 1 0 1 
0 1 0 1 0

48
'''