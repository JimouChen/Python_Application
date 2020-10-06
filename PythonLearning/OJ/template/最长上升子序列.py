"""
# @Time    :  2020/10/6
# @Author  :  Jimou Chen

"""
dp = [0 for i in  range(100)]
n = int(input())
nums = list(map(int, input().split()))
for i in range(0, n):
    dp[i] = 1
    for j in range(0, i):
        if nums[j] < nums[i]:
            dp[i] = max(dp[i], dp[j] + 1)


print(max(dp))

'''
5
1 5 2 4 3
'''