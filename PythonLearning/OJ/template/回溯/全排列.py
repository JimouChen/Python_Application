"""
# @Time    :  2020/11/25
# @Author  :  Jimou Chen
"""
'''
求1到n的全排列
用的思想是深搜dfs+回溯
'''
n = int(input())
box = [0 for _ in range(n + 1)]  # 存放排列后的数
flag = [0 for _ in range(n + 1)]  # 如果访问就为1


def dfs(step):
    if n == step:
        for i in range(n):
            print(box[i], end=' ')
        print()
        return

    for i in range(1, n + 1):
        if flag[i] == 0:
            box[step] = i  # 在该位置放置这个编号
            flag[i] = 1
            dfs(step + 1)
            flag[i] = 0


dfs(0)
