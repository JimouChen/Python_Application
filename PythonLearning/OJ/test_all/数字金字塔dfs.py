"""
# @Time    :  2020/10/11
# @Author  :  Jimou Chen
"""
'''
会超时
'''


def dfs(x, y, now_sum):
    if x == n:
        global s_max
        if now_sum > s_max:
            s_max = now_sum
        return

    dfs(x + 1, y, now_sum + a[x + 1][y])
    dfs(x + 1, y + 1, now_sum + a[x + 1][y + 1])


s_max = 0
n = int(input())
a = [[0 for i in range(101)] for j in range(101)]

for i in range(n):
    l = list(map(int, input().split()))
    for j in range(len(l)):
        a[i][j] = l[j]

dfs(0, 0, a[0][0])
print(s_max)

'''
5
7
3 8
8 1 0
2 7 4 4
4 5 2 6 5

30
'''
