"""
# @Time    :  2020/12/15
# @Author  :  Jimou Chen
"""
# n = int(input())
# max_n = 1000
# sum_n = [0 for _ in range(n+1)]  # 前缀和
# dp = [[-1 for _ in range(n+1)] for _ in range(n+1)]
#
# if __name__ == '__main__':
#     # n = int(input())
#     a = list(map(int, input().split()))
#     a.insert(0, 0)
#     # 计算前缀和
#     for i in range(1, n + 1):
#         sum_n[i] = sum_n[i - 1] + a[i]
#
#     # 从区间为1到n遍历,计算dp[i][j]
#     for length in range(1, n + 1):
#         i = 1
#         while i + length - 1 <= n:
#             j = i + length - 1
#             if length == 1:
#                 dp[i][j] = 0
#             else:
#                 for k in range(i, j):
#                     temp = dp[i][k] + dp[k + 1][j] + sum_n[j] - sum_n[i - 1]
#                     if dp[i][j] == -1 or dp[i][j] > temp:
#                         dp[i][j] = temp
#             i += 1
#
#     print(dp[1][n])


# n = int(input())
# sum_ = [0 for _ in range(n + 1)]
# dp = [[-1 for _ in range(n + 1)] for _ in range(n + 1)]
# a = list(map(int, input().split()))
# a.insert(0, 0)
# for i in range(1, 1 + n):
#     sum_[i] = sum_[i - 1] + a[i]
#
#
# def dfs(i, j):
#     if dp[i][j] != -1:
#         return dp[i][j]
#     if i == j:
#         dp[i][j] = 0
#         return 0
#     for k in range(i, j):
#         temp = dfs(i, k) + dfs(k + 1, j) + sum_[j] - sum_[i - 1]
#         if dp[i][j] == -1 or dp[i][j] > temp:
#             dp[i][j] = temp
#
#     return dp[i][j]
#
#
# print(dfs(1, n))

n = int(input())
s = [0 for _ in range(n + 1)]
dp = [[999999 for _ in range(n + 5)] for _ in range(n + 5)]
a = list(map(int, input().split()))
a.insert(0, 0)
for i in range(1, n + 1):
    s[i] += s[i - 1] + a[i]
    dp[i][i] = 0

for len_ in range(1, n + 1):
    i = 1
    while i + len_ <= n + 1:
        j = i + len_ - 1
        for k in range(i, j):
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + s[j] - s[i - 1])
        i += 1

print(dp[1][n])
