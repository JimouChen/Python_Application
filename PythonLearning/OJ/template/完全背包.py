"""
# @Time    :  2020/10/18
# @Author  :  Jimou Chen
"""
n, m = map(int, input().split())
w = [0 for _ in range(0, n + 1)]
v = [0 for _ in range(0, n + 1)]

for i in range(1, n + 1):
    w[i], v[i] = map(int, input().split())

dp = [[0 for i in range(m + 1)] for j in range(n + 1)]

for i in range(1, n + 1):
    for j in range(1, m + 1):
        if w[i] > j:
            dp[i][j] = dp[i - 1][j]
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - w[i]] + v[i])  # 这句与01背包不同

print(dp[n][m])
