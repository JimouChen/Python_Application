"""
# @Time    :  2020/10/11
# @Author  :  Jimou Chen
"""
a = [[0 for i in range(102)] for j in range(102)]
f = [[0 for i in range(102)] for j in range(102)]
n = int(input())
for i in range(1, 1 + n):
    l = list(map(int, input().split()))
    for j in range(1, i+1):
        a[i][j] = l[j-1]

f[1][1] = a[1][1]
for i in range(2, n + 1):
    for j in range(1, i + 1):
        f[i][j] = max(f[i - 1][j - 1], f[i - 1][j]) + a[i][j]

res = 0
for i in range(1, n + 1):
    res = max(res, f[n][i])

print(res)
