"""
# @Time    :  2020/9/20
# @Author  :  Jimou Chen
"""
num, weight = map(int, input().split())

# 第一个下标的值设为0，方便
w = [0]
v = [0]
for i in range(num):
    each_w, each_v = map(int, input().split())
    w.append(each_w)
    v.append(each_v)

# 用一个二维列表存储d[i][j]
dp = []
for i in range(0, num + 1):
    temp = []
    for j in range(0, weight + 1):
        temp.append(0)
    dp.append(temp)

for i in range(1, num + 1):
    for j in range(1, weight + 1):
        if j < w[i]:
            dp[i][j] = dp[i - 1][j]
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i])

print(dp[num][weight])
