"""
# @Time    :  2020/10/18
# @Author  :  Jimou Chen
"""

dp = [0 for i in range(10000)]
dp[0] = 0
if __name__ == '__main__':
    # 货币种类和金额
    n = int(input())
    class_list = list(map(int, input().split()))
    money = int(input())

    for i in range(1, money+1):
        cost = 9999999
        for j in range(n):
            if i >= class_list[j]:
                cost = min(cost, dp[i - class_list[j]] + 1)
        dp[i] = cost

    print(dp[money])

'''
4
1 2 5 10
8
'''
